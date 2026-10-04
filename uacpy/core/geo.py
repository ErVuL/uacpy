"""Geographic helpers: coordinates, great circles and calendar dates.

A carrier stores a coordinate as it was given (:func:`as_coordinate` checks it,
either longitude sign convention within one full wrap) and a lookup wraps the
longitude with :func:`normalize_lon`. :func:`great_circle_km`,
:func:`central_angle`, :func:`geodesic_waypoints` and
:func:`great_circle_midpoint` measure along the sphere of radius
:data:`EARTH_RADIUS_M`; :func:`parse_date` turns every date form the data layer
and :class:`~uacpy.core.environment.Environment` accept into a UTC calendar
date.
"""

import datetime as _dt
from typing import Optional, Tuple, Union

import numpy as np

from uacpy.core.exceptions import ConfigurationError

__all__ = [
    'Coordinate', 'EARTH_RADIUS_M', 'EARTH_RADIUS_KM', 'as_coordinate',
    'normalize_lon', 'central_angle', 'great_circle_km', 'geodesic_waypoints',
    'great_circle_midpoint', 'parse_date',
]


Coordinate = Tuple[float, float]

# Mean Earth radius (IUGG R1) for spherical great-circle geodesy when
# sampling external geographic datasets (bathymetry transects, …).
EARTH_RADIUS_M = 6_371_008.8  # m

#: :data:`EARTH_RADIUS_M` in km, the radius every haversine here uses.
EARTH_RADIUS_KM = EARTH_RADIUS_M / 1000.0

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


def as_coordinate(point, label: Optional[str] = None) -> Coordinate:
    """Validate and unpack a ``(lat, lon)`` coordinate pair as floats.

    The data layer takes a single ``point`` tuple everywhere, so this is the
    shared guard against the easy mistakes of passing two bare scalars / a
    single number, a non-finite (``NaN`` / ``inf``) coordinate, or a latitude
    outside ``[-90, 90]`` — each turns into a typed, actionable
    :class:`ConfigurationError` here rather than a cryptic unpack ``TypeError``
    or a downstream ``"cannot convert float NaN to integer"`` deeper in a
    fetcher's grid-index maths.

    The longitude is returned as given, in either sign convention up to one
    full wrap (``|lon| <= 360``); a lookup that needs ``[-180, 180)`` wraps
    it with :func:`normalize_lon`. ``label``, when given, starts every
    refusal (``"Environment: location"``).
    """
    prefix = f"{label}: " if label else ""
    try:
        lat, lon = point
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"{prefix}expected a (lat, lon) coordinate pair; got {point!r}.",
            remediation="Pass a 2-tuple of degrees, e.g. (43.2, 7.5).",
        ) from None
    if not (np.isfinite(lat) and np.isfinite(lon)):
        raise ConfigurationError(
            f"{prefix}coordinate must be finite; got (lat={lat}, lon={lon}).",
            remediation="Pass finite degrees, e.g. (43.2, 7.5).",
        )
    if not -90.0 <= lat <= 90.0:
        raise ConfigurationError(
            f"{prefix}latitude must be in [-90, 90] degrees; got {lat}.",
            remediation="Pass a latitude within [-90, 90].",
        )
    if not -360.0 <= lon <= 360.0:
        raise ConfigurationError(
            f"{prefix}longitude must be in [-360, 360] degrees; got {lon}.",
            remediation="Pass the longitude in either sign convention, "
                        "within one full wrap.",
        )
    return lat, lon


def normalize_lon(lon: float) -> float:
    """Wrap a longitude (degrees) into ``[-180, 180)``.

    Callers may pass longitude in either ``[-180, 180]`` or ``[0, 360]``;
    every source normalizes through here so the same physical point yields the
    same result regardless of convention (and dateline values stay in range).
    """
    return ((float(lon) + 180.0) % 360.0) - 180.0


def central_angle(start: Coordinate, end: Coordinate) -> float:
    """Great-circle central angle (radians) between two ``(lat, lon)`` points
    — :func:`great_circle_km` in radians, with the scalar coordinate checks."""
    lat1, lon1 = as_coordinate(start)
    lat2, lon2 = as_coordinate(end)
    return float(great_circle_km(lat1, lon1, lat2, lon2) / EARTH_RADIUS_KM)


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


def great_circle_midpoint(start: Coordinate, end: Coordinate) -> Coordinate:
    """The point halfway along the great circle from ``start`` to ``end``.

    The normalised sum of the endpoints' unit vectors, the point
    :func:`geodesic_waypoints` places at the middle of the path. Coincident
    endpoints return that point as given; endpoints antipodal to within
    :data:`_ANTIPODAL_TOL_RAD`, which no single great circle joins, are
    refused.
    """
    lat1, lon1 = as_coordinate(start)
    lat2, lon2 = as_coordinate(end)
    if (lat1, lon1) == (lat2, lon2):
        return lat1, lon1
    if np.pi - central_angle(start, end) < _ANTIPODAL_TOL_RAD:
        raise ConfigurationError(
            "great_circle_midpoint: the endpoints are antipodal, so no single "
            "great circle joins them.",
            remediation="Move an endpoint away from the other's antipode.")
    la1, lo1, la2, lo2 = np.radians([lat1, lon1, lat2, lon2])
    x = np.cos(la1) * np.cos(lo1) + np.cos(la2) * np.cos(lo2)
    y = np.cos(la1) * np.sin(lo1) + np.cos(la2) * np.sin(lo2)
    z = np.sin(la1) + np.sin(la2)
    return (float(np.degrees(np.arctan2(z, np.hypot(x, y)))),
            float(np.degrees(np.arctan2(y, x))))


def _to_utc_date(value: _dt.datetime) -> _dt.date:
    """Calendar date of ``value`` in UTC (naive input is taken as UTC)."""
    if value.tzinfo is not None:
        value = value.astimezone(_dt.timezone.utc)
    return value.date()


def parse_date(date: Union[str, _dt.date],
               label: Optional[str] = None) -> _dt.date:
    """Parse a calendar date into a :class:`datetime.date`.

    Accepts an ISO-8601 string (``'YYYY-MM-DD'`` or a full datetime string), a
    ``datetime.date`` / ``datetime.datetime``, or a ``numpy.datetime64`` (the
    type an xarray/pandas time coordinate yields, read as UTC like every
    naive value). A **timezone-aware** value is
    converted to UTC before the date is taken, because every dataset this
    feeds is indexed in UTC and the month selects the climatology slice.
    Raises a typed
    :class:`ConfigurationError` on a malformed value rather than letting a bare
    ``ValueError`` escape; ``label``, when given, starts the refusal.
    """
    prefix = f"{label}: " if label else ""
    if isinstance(date, _dt.datetime):
        # A tz-aware instant is converted to UTC first. Every dataset this
        # feeds (WOA23, NSIDC, Copernicus) is indexed in UTC, and the month
        # selects the climatology slice — so taking the local calendar date
        # picks the wrong month either side of midnight UTC.
        if date.tzinfo is not None:
            date = date.astimezone(_dt.timezone.utc)
        return date.date()
    if isinstance(date, _dt.date):
        return date
    if isinstance(date, np.datetime64):
        if np.isnat(date):
            raise ConfigurationError(
                f"{prefix}date is NaT (not a time).",
                remediation="Pass e.g. '2026-06-14' or a datetime.date.")
        return _dt.date.fromisoformat(str(date.astype('datetime64[D]')))
    if isinstance(date, str):
        for parse in (_dt.date.fromisoformat,
                      lambda s: _to_utc_date(_dt.datetime.fromisoformat(s))):
            try:
                return parse(date)
            except ValueError:
                continue
        raise ConfigurationError(
            f"{prefix}could not parse date {date!r}; expected ISO-8601 "
            f"(YYYY-MM-DD).",
            remediation="Pass e.g. '2026-06-14' or a datetime.date.",
        )
    raise ConfigurationError(
        f"{prefix}date must be an ISO-8601 string, datetime.date or "
        f"numpy.datetime64; "
        f"got {type(date).__name__}.",
        remediation="Pass e.g. '2026-06-14' or a datetime.date.",
    )
