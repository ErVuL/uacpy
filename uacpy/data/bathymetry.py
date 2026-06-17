"""GEBCO bathymetry fetch — GPS coordinates → ``Environment``-ready depth.

Phase 1 of the on-demand external-data layer: turn geographic coordinates
into a water depth (m, positive down) ready to hand to
``Environment(bathymetry=...)`` — either a single depth for one point, or a
range-dependent ``(N, 2)`` ``[range_m, depth_m]`` transect sampled along the
great-circle path between two points.

Depths come from the GEBCO_2020 global grid (~450 m resolution), served as
JSON by the public OpenTopoData API
(https://www.opentopodata.org/datasets/gebco2020/). No API key is needed;
the public host is rate-limited (≤100 locations per request, ≤1 request/s,
≤1000 requests/day). Point a ``base_url`` at a self-hosted OpenTopoData
instance to lift those limits.

Bathymetry is static in time, so these fetches take coordinates only — the
``date`` axis enters the data layer at the sound-speed stage, not here.
"""

import json
import urllib.parse
from typing import List, Tuple, Union

import numpy as np

from uacpy.core.constants import EARTH_RADIUS_M
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data._geo import Coordinate, as_coordinate, normalize_lon
from uacpy.data._http import http_get
from uacpy._log import log_message

__all__ = ['fetch_point_depth', 'fetch_transect', 'fetch_grid',
           'transect_length']

DEFAULT_BASE_URL = 'https://api.opentopodata.org/v1'
DEFAULT_DATASET = 'gebco2020'
MAX_LOCATIONS_PER_REQUEST = 100  # OpenTopoData public-host limit
MAX_GRID_REQUESTS = 100          # safety cap for fetch_grid (≤10 000 points)

BATHY_SOURCES = ('api', 'gmrt', 'local')


def _check_source(source):
    if source not in BATHY_SOURCES:
        raise ConfigurationError(
            f"bathymetry source must be one of {BATHY_SOURCES}; got {source!r}.",
            remediation="Use 'api' (OpenTopoData/GEBCO), 'gmrt' (GMRT multibeam, "
                        "higher-res live), or 'local' (offline GEBCO grid).",
        )


def fetch_point_depth(
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
    source : {'api', 'gmrt', 'local'}, optional
        ``'api'`` (default) queries OpenTopoData/GEBCO online; ``'gmrt'`` queries
        the GMRT multibeam synthesis (higher resolution where surveyed, CC-BY,
        live); ``'local'`` samples the install-time GEBCO grid (offline, no rate
        limit; see ``install.sh --data gebco``).
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
    depths = _fetch_depths(
        [as_coordinate(point)], dataset=dataset, base_url=base_url,
        timeout=timeout, verbose=verbose,
    )
    return float(depths[0])


def fetch_transect(
    start: Coordinate,
    end: Coordinate,
    *,
    n_points: int = 50,
    source: str = 'api',
    dataset: str = DEFAULT_DATASET,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    verbose: Union[bool, str] = False,
) -> np.ndarray:
    """Range-dependent bathymetry along the great-circle ``start``→``end``.

    Samples ``n_points`` evenly spaced (in distance) along the geodesic and
    returns an ``(n_points, 2)`` array of ``[range_m, depth_m]`` with
    ``range`` measured from ``start`` — exactly the shape consumed by
    ``Environment(bathymetry=...)`` for range-dependent runs.

    Parameters
    ----------
    start, end : (lat, lon)
        Endpoint coordinates in decimal degrees (WGS84).
    n_points : int, optional
        Number of samples (≥2). Default 50.
    dataset, base_url, timeout, verbose
        See :func:`fetch_point_depth`.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_points, 2)``: column 0 range (m), column 1 depth (m).

    Raises
    ------
    ConfigurationError
        ``n_points < 2`` or the two endpoints coincide.
    DataFetchError
        The service fails, or any sampled point falls on land.
    """
    _check_source(source)
    if n_points < 2:
        raise ConfigurationError(
            f"fetch_transect: n_points must be >= 2, got {n_points}.",
            remediation="Pass n_points=2 or more to define a transect.",
        )

    lats, lons, ranges_m = _geodesic_waypoints(start, end, n_points)
    log_message(
        'bathymetry', f"sampling {n_points} GEBCO depths along "
        f"{ranges_m[-1] / 1000:.1f} km transect", verbose=verbose,
    )
    if source == 'local':
        from uacpy.data import gebco_local
        depths = gebco_local.depths_along(lats, lons)
    elif source == 'gmrt':
        from uacpy.data import gmrt_live
        depths = gmrt_live.depths_along(lats, lons, timeout=timeout, verbose=verbose)
    else:
        depths = _fetch_depths(
            list(zip(lats, lons)), dataset=dataset, base_url=base_url,
            timeout=timeout, verbose=verbose,
        )
    return np.column_stack([ranges_m, depths])


def fetch_grid(
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
            f"fetch_grid: n_lat and n_lon must be >= 2; got {n_lat}, {n_lon}.")
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
    n_requests = -(-(n_lat * n_lon) // MAX_LOCATIONS_PER_REQUEST)   # ceil-div
    if n_requests > MAX_GRID_REQUESTS:
        raise ConfigurationError(
            f"fetch_grid: {n_lat}×{n_lon} = {n_lat * n_lon} points needs "
            f"{n_requests} requests (> {MAX_GRID_REQUESTS}).",
            remediation="Use a coarser grid, or a self-hosted instance via base_url=.",
        )
    lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    lons = np.linspace(lon_range[0], lon_range[1], n_lon)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    coords = list(zip(lat_mesh.ravel(), lon_mesh.ravel()))
    log_message('bathymetry', f"GEBCO grid {n_lat}×{n_lon} over "
                f"lat{lat_range} lon{lon_range}", verbose=verbose)
    elev = _fetch_elevations(coords, dataset=dataset, base_url=base_url,
                             timeout=timeout, verbose=verbose)
    depth = np.where(elev < 0.0, -elev, np.nan).reshape(n_lat, n_lon)
    return lats, lons, depth


def _central_angle(start: Coordinate, end: Coordinate) -> float:
    """Great-circle central angle (radians) between two ``(lat, lon)`` points
    (haversine; numerically stable for small separations)."""
    lat1, lon1 = np.radians(as_coordinate(start))
    lat2, lon2 = np.radians(as_coordinate(end))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * np.arcsin(np.sqrt(a)))


def transect_length(start: Coordinate, end: Coordinate) -> float:
    """Great-circle transect length (m) from ``start`` to ``end`` ``(lat, lon)``.

    Uses the same spherical geodesic as the range-dependent fetchers, so it
    equals the maximum range of the fetched bathymetry / SSP transect — size a
    receiver grid directly::

        L = uacpy.data.transect_length(A, B)
        rcv = uacpy.Receiver(depths=..., ranges=np.linspace(0.0, L, n))

    Returns ``0.0`` for coincident endpoints.
    """
    return _central_angle(start, end) * EARTH_RADIUS_M


def _geodesic_waypoints(
    start: Coordinate, end: Coordinate, n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evenly spaced great-circle waypoints between two ``(lat, lon)``.

    Returns ``(lats_deg, lons_deg, ranges_m)`` where ``ranges_m`` is the
    cumulative spherical surface distance from ``start`` (0 at the first
    point, total length at the last). Spherical-Earth slerp — accurate to a
    few parts in 10³ versus the WGS84 ellipsoid, ample for sampling a grid
    of ~450 m resolution.
    """
    lat1, lon1 = np.radians(as_coordinate(start))
    lat2, lon2 = np.radians(as_coordinate(end))

    ang = _central_angle(start, end)              # shared geodesic (haversine)
    if ang == 0.0:
        raise ConfigurationError(
            "fetch_transect: start and end coordinates coincide.",
            remediation="Use fetch_point_depth for a single location, or "
                        "pass distinct endpoints.",
        )

    f = np.linspace(0.0, 1.0, n_points)
    sin_ang = np.sin(ang)
    A = np.sin((1 - f) * ang) / sin_ang
    B = np.sin(f * ang) / sin_ang
    x = A * np.cos(lat1) * np.cos(lon1) + B * np.cos(lat2) * np.cos(lon2)
    y = A * np.cos(lat1) * np.sin(lon1) + B * np.cos(lat2) * np.sin(lon2)
    z = A * np.sin(lat1) + B * np.sin(lat2)
    lats = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lons = np.degrees(np.arctan2(y, x))

    ranges_m = f * ang * EARTH_RADIUS_M
    return lats, lons, ranges_m


def _fetch_elevations(
    coords: List[Coordinate],
    *,
    dataset: str,
    base_url: str,
    timeout: float,
    verbose: Union[bool, str],
) -> np.ndarray:
    """Raw GEBCO elevations (m, positive up) for ``coords``, chunked to the
    OpenTopoData per-call cap. Land is positive; ocean negative."""
    elevations: List[float] = []
    for i in range(0, len(coords), MAX_LOCATIONS_PER_REQUEST):
        chunk = coords[i:i + MAX_LOCATIONS_PER_REQUEST]
        elevations.extend(
            _request_chunk(chunk, dataset=dataset, base_url=base_url,
                           timeout=timeout, verbose=verbose)
        )
    return np.asarray(elevations, dtype=float)


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
