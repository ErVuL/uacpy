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
    'DEFAULT_MAX_TRANSECT_POINTS', 'checked_max_points',
    'checked_n_points',
    'depth_to_pressure_dbar', 'pressure_dbar_to_depth',
    'insitu_from_potential',
]

#: Default ceiling on the number of points sampled along a transect *before*
#: the ``'auto'`` reduction step — i.e. the fetch budget. The reduced grid is
#: never larger than this (and usually much smaller after collapsing
#: duplicates). At OpenTopoData's ≤100 locations/request this is ≤10 requests
#: for a full-native bathy transect. Override via ``max_points=`` on the
#: fetchers / fetch_environment.
DEFAULT_MAX_TRANSECT_POINTS = 1000

Coordinate = Tuple[float, float]


def checked_max_points(max_points, caller: str) -> int:
    """Validate a transect fetch budget: an integer of at least 2.

    ``n_points`` is guarded at >= 2 wherever it is accepted, but the cap it is
    reduced against was not, and ``'auto'`` resolves straight to the cap:
    ``max_points=1`` produced a one-waypoint "transect" with
    ``ranges_m=[0.0]``, ``max_points=0`` an empty one, and a negative value an
    untyped ``ValueError`` out of ``np.linspace``. Two points are the fewest
    that define a path.
    """
    try:
        value = int(max_points)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"{caller}: max_points must be an integer >= 2, got "
            f"{max_points!r}.",
            remediation="Pass max_points>=2, the transect fetch budget.",
        ) from None
    if value < 2:
        raise ConfigurationError(
            f"{caller}: max_points must be >= 2, got {max_points}.",
            remediation="Pass max_points>=2; fewer than two waypoints is not "
                        "a transect.",
        )
    return value


def checked_n_points(n_points, label: str, *, allow_auto: bool = False):
    """Validate a transect sample count: an integer of at least 2.

    The eight transect fetchers each rolled their own guard and behaved five
    different ways on the same input. ``n_points=1`` was a typed
    ``ConfigurationError`` at six sites and a silent coercion to 2 at
    ``range_dependent_bottom_along``; ``n_points=2.7`` was silently truncated
    to 2 at two sites, an untyped ``TypeError`` at four and typed only at
    ``bathy_transect_plan``; ``n_points='x'`` was an untyped ``ValueError``
    almost everywhere. Two waypoints are the fewest that define a path, and a
    fractional or non-numeric count is a caller mistake rather than something
    to round — so all of them come here now, and ``bathy_transect_plan``'s
    behaviour (the one that was already right) is the behaviour they share.

    Parameters
    ----------
    n_points : int or str
        The caller's value, unvalidated.
    label : str
        Caller name for the message, e.g. ``'fetch_wind_transect'``.
    allow_auto : bool, optional
        Accept the string ``'auto'`` and return it unchanged, for the fetchers
        that resolve a native sample count themselves. Default ``False``.

    Returns
    -------
    int or str
        The count as an ``int``, or ``'auto'`` when ``allow_auto`` admitted it.

    Raises
    ------
    ConfigurationError
        ``n_points`` is below 2, not a whole number, or not a number at all.
    """
    if allow_auto and isinstance(n_points, str) and n_points == 'auto':
        return 'auto'
    forms = ("an integer >= 2, or 'auto'" if allow_auto else "an integer >= 2")
    remediation = ("Pass n_points as an int (e.g. n_points=50)"
                   + (" or n_points='auto'." if allow_auto else "."))
    msg = (f"{label}: n_points={n_points!r} is not a sample count. "
           f"Valid forms: {forms}.")
    try:
        value = int(n_points)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(msg, remediation=remediation) from exc
    if value != n_points:            # 2.7 was silently truncated to 2
        raise ConfigurationError(msg, remediation=remediation)
    if value < 2:
        raise ConfigurationError(
            f"{label}: n_points must be >= 2, got {n_points}.",
            remediation="Pass n_points>=2"
                        + (" or 'auto'; " if allow_auto else "; ")
                        + "fewer than two waypoints is not a transect.",
        )
    return value


#: How close to antipodal (radians of central angle short of π) a pair of
#: endpoints may be before :func:`geodesic_waypoints` refuses them. The slerp
#: divides by ``sin(ang)``, and the haversine central angle saturates at
#: exactly π once the endpoints are within ~0.1 m of antipodal, so past this
#: point the waypoints stop lying on the path ``ranges_m`` reports. Measured
#: disagreement between a waypoint's great-circle distance from ``start`` and
#: its reported range: 0.13 m at π − ang = 1e-5, 142 m at 1e-6, 37 km at 1e-7
#: and 1083 km at exactly π. 1e-6 rad is ~6 m of antipodal offset, so no real
#: transect is refused.
_ANTIPODAL_TOL_RAD = 1e-6

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
    — :func:`great_circle_km` in radians, with the scalar coordinate checks."""
    lat1, lon1 = as_coordinate(start)
    lat2, lon2 = as_coordinate(end)
    return float(great_circle_km(lat1, lon1, lat2, lon2) / EARTH_RADIUS_KM)


def require_month(month, who: str) -> int:
    """``month`` as an int in 1-12, or ``ConfigurationError``: a bool, a
    fraction (``6.7``) or a non-number is refused rather than truncated to a
    month the caller never asked for."""
    if isinstance(month, (bool, np.bool_)):
        raise ConfigurationError(f"{who}: month={month!r} is a bool, not a month (1-12).")
    try:
        value = float(month)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"{who}: month={month!r} is not a number; pass an integer 1-12.") from None
    if not value.is_integer() or not 1 <= value <= 12:
        raise ConfigurationError(
            f"{who}: month={month!r} is not an integer month 1-12; a fraction "
            f"would be truncated to a month you did not ask for.")
    return int(value)


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

    Coincident endpoints, and endpoints antipodal to within
    :data:`_ANTIPODAL_TOL_RAD` (where infinitely many great circles join them
    and the slerp's ``1/sin(ang)`` returns waypoints that do not lie on the
    reported ranges), raise :class:`ConfigurationError`.
    """
    lat1, lon1 = np.radians(as_coordinate(start))
    lat2, lon2 = np.radians(as_coordinate(end))

    ang = central_angle(start, end)               # shared geodesic (haversine)
    if ang == 0.0:
        raise ConfigurationError(
            "geodesic_waypoints: start and end coordinates coincide.",
            remediation="Use a single-point fetch, or pass distinct endpoints.",
        )
    if np.pi - ang < _ANTIPODAL_TOL_RAD:
        raise ConfigurationError(
            f"geodesic_waypoints: the endpoints are antipodal to within "
            f"{(np.pi - ang) * EARTH_RADIUS_M:.1f} m, so no single great "
            f"circle joins them.",
            remediation="Split the path into two transects through an "
                        "intermediate waypoint, or move an endpoint away from "
                        "the other's antipode.",
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


def pressure_dbar_to_depth(pres_dbar, lat) -> np.ndarray:
    """Pressure (dbar) → depth (m): Newton inversion of
    :func:`depth_to_pressure_dbar`.

    Pressure-indexed sources (Argo reports pressure) need depth for a
    ``SoundSpeedProfile``, and only the depth → pressure direction has a
    closed form. The derivative is taken as a central difference on
    :func:`depth_to_pressure_dbar` rather than analytically, so the Leroy &
    Parthiot coefficients live in exactly one place; a 1 m step is safe
    because ``h(z)`` is a smooth quartic whose curvature over a metre is
    negligible against its ~1 dbar/m slope.
    """
    p = np.asarray(pres_dbar, dtype=float)
    z = p * 0.9905                                   # ~1 m per dbar initial guess
    for _ in range(5):
        f = depth_to_pressure_dbar(z, lat) - p
        df = (depth_to_pressure_dbar(z + 1.0, lat)
              - depth_to_pressure_dbar(z - 1.0, lat)) / 2.0
        z = z - f / df
    return z


def _adiabatic_gradient(sal, temp, pres):
    """Adiabatic temperature gradient (°C/dbar), Bryden (1973).

    The polynomial as published in UNESCO Technical Papers in Marine Science
    44 (1983), Fofonoff & Millard, routine ``ATG``. Salinity is practical
    salinity (PSS-78), ``temp`` in-situ °C, ``pres`` in decibars. The paper's
    check value ``ATG(S=40, T=40, P=10000) = 3.255976e-4 °C/dbar`` is pinned
    in the tests.
    """
    ds = np.asarray(sal, dtype=float) - 35.0
    t = np.asarray(temp, dtype=float)
    p = np.asarray(pres, dtype=float)
    return ((((-2.1687e-16 * t + 1.8676e-14) * t - 4.6206e-13) * p
             + ((2.7759e-12 * t - 1.1351e-10) * ds
                + ((-5.4481e-14 * t + 8.733e-12) * t - 6.7795e-10) * t
                + 1.8741e-8)) * p
            + (-4.2393e-8 * t + 1.8932e-6) * ds
            + ((6.6228e-10 * t - 6.836e-8) * t + 8.5258e-6) * t + 3.5803e-5)


def _shift_adiabatically(sal, temp, pres_from, pres_to):
    """Move a water parcel adiabatically from ``pres_from`` to ``pres_to``.

    Fourth-order Runge-Kutta integration of :func:`_adiabatic_gradient` over
    the pressure interval, in the coefficient form of UNESCO 44's ``THETA``
    (Fofonoff 1977). One RK4 step spans the whole interval, which is what the
    reference routine does and what its check value
    ``THETA(S=40, T=40, P=10000, Pr=0) = 36.89073 °C`` certifies; the gradient
    is a slowly varying polynomial, so the round trip closes to better than
    2e-4 °C over the full oceanic range (both pinned in the tests).
    """
    p = np.asarray(pres_from, dtype=float)
    t = np.asarray(temp, dtype=float)
    h = np.asarray(pres_to, dtype=float) - p
    xk = h * _adiabatic_gradient(sal, t, p)
    t = t + 0.5 * xk
    q = xk
    p = p + 0.5 * h
    xk = h * _adiabatic_gradient(sal, t, p)
    t = t + 0.29289322 * (xk - q)
    q = 0.58578644 * xk + 0.121320344 * q
    xk = h * _adiabatic_gradient(sal, t, p)
    t = t + 1.707106781 * (xk - q)
    q = 3.414213562 * xk - 4.121320344 * q
    p = p + 0.5 * h
    xk = h * _adiabatic_gradient(sal, t, p)
    return t + (xk - 2.0 * q) / 6.0


def insitu_from_potential(sal, theta, pres) -> np.ndarray:
    """Potential temperature (°C, referenced to the surface) → in-situ °C.

    Ocean models report ``thetao``, the temperature a parcel *would* have if
    brought adiabatically to 0 dbar; the sound-speed equations (UNESCO,
    Del Grosso) want the temperature the parcel actually has at depth. The two
    are the same at the surface and diverge with pressure: a parcel is warmed
    by compression, so in-situ is always the warmer of the pair below 0 dbar.

    Ignoring the difference is a deep-water error, not a uniform one. At
    S=34.7 the in-situ excess and the sound-speed error it costs are

    ======  ==========  =======  ==============
    depth   theta (°C)  ΔT (°C)  Δc (m/s)
    ======  ==========  =======  ==============
    2000 m         2.5    0.149  +0.64
    5000 m         1.5    0.462  +1.97
    10000 m        1.2    1.286  +4.98
    ======  ==========  =======  ==============

    (UNESCO; Del Grosso agrees within 0.2 m/s.) Climatology and float sources
    are unaffected — WOA23's ``t_an`` and Argo's ``TEMP`` are already in-situ.

    Parameters
    ----------
    sal : array_like
        Practical salinity (PSS-78). Conserved by the adiabatic shift.
    theta : array_like
        Potential temperature (°C), referenced to 0 dbar.
    pres : array_like
        In-situ pressure (dbar).

    Returns
    -------
    numpy.ndarray
        In-situ temperature (°C).
    """
    return np.asarray(
        _shift_adiabatically(sal, theta, 0.0, pres), dtype=float)
