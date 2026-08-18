"""Small shared geographic helpers for the data layer."""

from typing import Tuple

import numpy as np

from uacpy.core.constants import EARTH_RADIUS_M
from uacpy.core.exceptions import ConfigurationError

__all__ = [
    'Coordinate', 'as_coordinate', 'normalize_lon', 'lon_linspace',
    'EARTH_RADIUS_KM', 'central_angle', 'great_circle_km', 'geodesic_waypoints',
    'nearest_indices', 'ring_offsets', 'run_representative_indices',
    'run_boundary_indices',
    'DEFAULT_MAX_TRANSECT_POINTS',
    'depth_to_pressure_dbar',
]

#: Default ceiling on the number of points sampled along a transect *before*
#: the ``'auto'`` reduction step — i.e. the fetch budget. The reduced grid is
#: never larger than this (and usually much smaller after collapsing
#: duplicates). At OpenTopoData's ≤100 locations/request this is ≤10 requests
#: for a full-native bathy transect. Override via ``max_points=`` on the
#: fetchers / fetch_environment.
DEFAULT_MAX_TRANSECT_POINTS = 1000

Coordinate = Tuple[float, float]

#: Spherical-Earth radius in km, derived from the single source of truth in
#: ``core.constants`` so every haversine in the data layer shares one value.
EARTH_RADIUS_KM = EARTH_RADIUS_M / 1000.0


def as_coordinate(point) -> Coordinate:
    """Validate and unpack a ``(lat, lon)`` coordinate pair as floats.

    The data layer takes a single ``point`` tuple everywhere, so this is the
    shared guard against the easy mistakes of passing two bare scalars / a
    single number, a non-finite (``NaN`` / ``inf``) coordinate, or a latitude
    outside ``[-90, 90]`` — each turns into a typed, actionable
    :class:`ConfigurationError` here rather than a cryptic unpack ``TypeError``
    or a downstream ``"cannot convert float NaN to integer"`` deeper in a
    fetcher's grid-index maths.
    """
    try:
        lat, lon = point
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"expected a (lat, lon) coordinate pair; got {point!r}.",
            remediation="Pass a 2-tuple of degrees, e.g. (43.2, 7.5).",
        ) from None
    if not (np.isfinite(lat) and np.isfinite(lon)):
        raise ConfigurationError(
            f"coordinate must be finite; got (lat={lat}, lon={lon}).",
            remediation="Pass finite degrees, e.g. (43.2, 7.5).",
        )
    if not -90.0 <= lat <= 90.0:
        raise ConfigurationError(
            f"latitude must be in [-90, 90] degrees; got {lat}.",
            remediation="Pass a latitude within [-90, 90] (longitude wraps).",
        )
    return lat, lon


def normalize_lon(lon: float) -> float:
    """Wrap a longitude (degrees) into ``[-180, 180)``.

    Callers may pass longitude in either ``[-180, 180]`` or ``[0, 360]``;
    every source normalizes through here so the same physical point yields the
    same result regardless of convention (and dateline values stay in range).
    """
    return ((float(lon) + 180.0) % 360.0) - 180.0


def lon_linspace(lon0: float, lon1: float, n: int) -> np.ndarray:
    """``n`` longitudes from ``lon0`` to ``lon1`` going **eastward**, wrapped to
    ``[-180, 180)``.

    A range whose end is west of its start (e.g. ``(179, -179)``) is taken to
    cross the antimeridian eastward, so it samples the short strip over 180°
    rather than sweeping the long way through 0°. This mirrors the great-circle
    transect path; a non-crossing range (``lon1 >= lon0``) is a plain linspace.
    """
    lon0, lon1 = float(lon0), float(lon1)
    if lon1 < lon0:
        lon1 += 360.0
    raw = np.linspace(lon0, lon1, int(n))
    wrapped = ((raw + 180.0) % 360.0) - 180.0
    # Keep a node that lies exactly on +180 as +180: wrapping it to -180 made
    # a full-globe axis non-monotonic with a duplicated first column. Exact
    # equality is the right test — only an exact +180 wraps to exactly -180,
    # while other multiples of 360 offset from it (-180, 540) stay -180.
    wrapped[raw == 180.0] = 180.0
    return wrapped


def central_angle(start: Coordinate, end: Coordinate) -> float:
    """Great-circle central angle (radians) between two ``(lat, lon)`` points
    (haversine; numerically stable for small separations)."""
    lat1, lon1 = np.radians(as_coordinate(start))
    lat2, lon2 = np.radians(as_coordinate(end))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * np.arcsin(np.sqrt(a)))


def great_circle_km(lat0, lon0, lat, lon):
    """Haversine great-circle distance (km) from ``(lat0, lon0)`` to ``(lat,
    lon)``. Vectorized over the second point; uses :data:`EARTH_RADIUS_KM`."""
    la0, lo0, la, lo = map(np.radians, (lat0, lon0, lat, lon))
    d = (np.sin((la - la0) / 2) ** 2
         + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(d))


def geodesic_waypoints(
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

    ang = central_angle(start, end)               # shared geodesic (haversine)
    if ang == 0.0:
        raise ConfigurationError(
            "geodesic_waypoints: start and end coordinates coincide.",
            remediation="Use a single-point fetch, or pass distinct endpoints.",
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


def nearest_indices(axis, queries) -> np.ndarray:
    """Nearest-node index into ``axis`` for each query, any axis orientation.

    A COARDS latitude axis is commonly stored **descending** (GMRT, EMODnet
    DTM), so a plain ``searchsorted`` (ascending-only, and an insertion index
    rather than the nearest node) is both biased and wrong. Sort once, bracket
    with ``searchsorted``, then pick the closer of the two neighbours, and map
    back to the original (possibly descending) ordering.
    """
    axis = np.asarray(axis, dtype=float)
    order = np.argsort(axis)
    sorted_axis = axis[order]
    pos = np.searchsorted(sorted_axis, queries)
    lo = np.clip(pos - 1, 0, sorted_axis.size - 1)
    hi = np.clip(pos, 0, sorted_axis.size - 1)
    pick_hi = np.abs(sorted_axis[hi] - queries) < np.abs(queries - sorted_axis[lo])
    nearest_sorted = np.where(pick_hi, hi, lo)
    return order[nearest_sorted]


def ring_offsets(radius: int) -> 'list[tuple[int, int]]':
    """``(d_row, d_col)`` offsets on the Chebyshev ring of the given radius,
    ordered nearest-first by squared Euclidean distance in cells.

    The expanding-ring neighbour search shared by the gridded climatology
    readers (WOA23's nearest-wet-cell fallback, NSIDC sea ice's
    nearest-observed-cell fallback): probe radius 1, then 2, … so the first
    hit is the closest usable cell.
    """
    out = [(d_row, d_col)
           for d_row in range(-radius, radius + 1)
           for d_col in range(-radius, radius + 1)
           if max(abs(d_row), abs(d_col)) == radius]
    return sorted(out, key=lambda o: o[0] ** 2 + o[1] ** 2)


def run_representative_indices(keys) -> 'list[int]':
    """Indices of one representative per maximal run of consecutive-equal keys.

    The reduction behind ``'auto'`` transect sampling for **interpolated**
    columns (the WOA23 SSP): probe a source's sample *identity* (e.g. the grid
    cell, or the nearest sample) at a fine set of waypoints, then keep one
    waypoint per distinct run. ``keys`` must be ``==``-comparable (tuples,
    scalars, or dataclasses); do not pass raw arrays. A carrier reconstructed
    by **nearest-node** lookup (categorical Surface/Bottom) uses
    :func:`run_boundary_indices` instead — a midpoint representative would
    displace each reconstructed transition to midway between run centres.

    Interior runs are represented by their **midpoint** (the centre of the
    range interval that sample covers). The **first and last** runs are
    anchored to the transect **endpoints** (indices ``0`` and ``n-1``) so the
    reduced range axis explicitly spans the full transect ``[0, L]`` rather
    than ``[first-cell-centre, last-cell-centre]``. A single run collapses to
    one representative at the start (range-independent transect).
    """
    runs = []
    i, n = 0, len(keys)
    while i < n:
        j = i
        while j + 1 < n and keys[j + 1] == keys[i]:
            j += 1
        runs.append((i, j))
        i = j + 1
    last = len(runs) - 1
    reps: list = []
    for k, (i, j) in enumerate(runs):
        if k == 0:
            reps.append(0)            # anchor transect start (range 0)
        elif k == last:
            reps.append(n - 1)        # anchor transect end (range L)
        else:
            reps.append((i + j) // 2)  # interior cell centre
    return reps


def run_boundary_indices(keys) -> 'list[int]':
    """Indices keeping both probe samples that bracket every change of key,
    plus the two transect endpoints.

    The reduction behind ``'auto'`` transect sampling for **categorical**
    carriers reconstructed by nearest-node lookup (the ice-canopy/open-water
    ``Surface``, the sediment-identity ``Bottom``): a nearest-node read places
    each transition midway between adjacent kept samples, so keeping the last
    sample of one run and the first of the next pins the reconstructed
    transition to within half a probe step of the boundary the probe observed.
    ``keys`` must be ``==``-comparable, as in
    :func:`run_representative_indices`; a single run collapses to one
    representative at the start (range-independent transect).
    """
    n = len(keys)
    if n == 0:
        return []
    out = [0]
    for i in range(1, n):
        if not keys[i] == keys[i - 1]:
            if out[-1] != i - 1:
                out.append(i - 1)   # last sample of the run ending at i-1
            out.append(i)           # first sample of the run starting at i
    if len(out) == 1:
        return out                  # single run: one representative (start)
    if out[-1] != n - 1:
        out.append(n - 1)           # anchor transect end (range L)
    return out


def depth_to_pressure_dbar(depth_m, latitude_deg) -> np.ndarray:
    """Depth (m) → pressure (dbar), Leroy & Parthiot (1998) standard ocean.

    JASA 103(3), 1346-1352, eqs. (8)-(11) — ``h(Z,phi) = h(Z,45)·k(Z,phi)``
    with ``Z`` in metres and ``h`` in MPa, hence the ×100 to dbar. The authors
    give the fit as accurate to ±500 Pa over the whole depth/latitude range.

    This is the standard ocean: the per-region geopotential corrective term
    ``delta_h_i`` of their eq. (12) / Table II is not applied, so a basin with
    a strongly non-standard T/S profile (Mediterranean, Baltic, Black Sea)
    carries that residual. ``soundspeed_*`` expect dbar.
    """
    z = np.asarray(depth_m, dtype=float)
    phi = np.radians(latitude_deg)
    g_phi = 9.7803 * (1 + 5.3e-3 * np.sin(phi) ** 2)
    h45 = (1.00818e-2 * z + 2.465e-8 * z ** 2
           - 1.25e-13 * z ** 3 + 2.8e-19 * z ** 4)          # MPa
    k = (g_phi - 2e-5 * z) / (9.80612 - 2e-5 * z)
    return h45 * k * 100.0                                   # MPa → dbar
