"""Tests for the shared geographic helpers (uacpy.data._geo)."""

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError
from uacpy.data._geo import as_coordinate, normalize_lon


@pytest.mark.parametrize('given,expected', [
    (0.0, 0.0),
    (-140.0, -140.0),
    (220.0, -140.0),     # 0–360 convention
    (360.0, 0.0),
    (180.0, -180.0),     # +180 wraps to -180
    (190.0, -170.0),     # just past dateline
    (-200.0, 160.0),
])
def test_normalize_lon(given, expected):
    assert normalize_lon(given) == pytest.approx(expected)


def test_as_coordinate_accepts_pairs():
    assert as_coordinate((43.2, 7.5)) == (43.2, 7.5)
    assert as_coordinate([1, 2]) == (1.0, 2.0)        # list, ints → floats


@pytest.mark.parametrize('bad', [43.2, (1.0,), (1, 2, 3), 'ab', None])
def test_as_coordinate_rejects_non_pairs(bad):
    with pytest.raises(ConfigurationError, match='coordinate pair'):
        as_coordinate(bad)


@pytest.mark.parametrize('bad', [
    (float('nan'), 0.0), (0.0, float('nan')),
    (float('inf'), 0.0), (0.0, float('-inf')),
])
def test_as_coordinate_rejects_non_finite(bad):
    # float() accepts NaN/inf, so without an explicit finiteness check they
    # reach a fetcher's grid-index arithmetic and surface there as a raw
    # "cannot convert float NaN to integer" far from the bad input.
    with pytest.raises(ConfigurationError, match='finite'):
        as_coordinate(bad)


@pytest.mark.parametrize('lat', [95.0, -91.0, 90.001])
def test_as_coordinate_rejects_out_of_range_latitude(lat):
    with pytest.raises(ConfigurationError, match=r'\[-90, 90\]'):
        as_coordinate((lat, 0.0))


def test_as_coordinate_allows_unwrapped_longitude():
    # Longitude is cyclic and normalized downstream, so it is left as-is here.
    assert as_coordinate((0.0, 200.0)) == (0.0, 200.0)
    assert as_coordinate((-89.9, -181.0)) == (-89.9, -181.0)


def test_transect_length_pins_the_documented_geodesic():
    # Guide endpoints (48.2, -8.0) → (45.6, -6.2): the spherical haversine on
    # the shared EARTH_RADIUS_M (6 371 008.8 m) gives 319 797.92 m. The WGS84
    # ellipsoidal length is 319 906.6 m, 0.034 % longer — within the
    # documented few-parts-in-10³ spherical approximation.
    from uacpy.data.bathymetry import transect_length
    A, B = (48.2, -8.0), (45.6, -6.2)
    assert transect_length(A, B) == pytest.approx(319797.9, abs=0.05)
    assert transect_length(B, A) == transect_length(A, B)
    assert transect_length(A, A) == 0.0


def test_geodesic_waypoints_round_trip_the_endpoints():
    # First/last waypoints equal the requested endpoints, and the range axis
    # runs 0 → transect_length, strictly increasing.
    from uacpy.data._geo import geodesic_waypoints
    from uacpy.data.bathymetry import transect_length
    A, B = (48.2, -8.0), (45.6, -6.2)
    lats, lons, ranges = geodesic_waypoints(A, B, 7)
    assert (lats[0], lons[0]) == pytest.approx(A)
    assert (lats[-1], lons[-1]) == pytest.approx(B)
    assert ranges[0] == 0.0
    assert ranges[-1] == pytest.approx(transect_length(A, B))
    assert np.all(np.diff(ranges) > 0)


def test_env_max_range_matches_the_transect_length():
    # A bathymetry sampled on the A→B geodesic ranges makes env.max_range the
    # transect length, and env.transect carries the two endpoints.
    import uacpy
    from uacpy.data._geo import geodesic_waypoints
    from uacpy.data.bathymetry import transect_length
    A, B = (48.2, -8.0), (45.6, -6.2)
    _, _, ranges = geodesic_waypoints(A, B, 5)
    env = uacpy.Environment(
        bathymetry=np.column_stack([ranges, np.full(ranges.size, 4000.0)]),
        ssp=1500.0, transect=(A, B))
    assert env.max_range == pytest.approx(transect_length(A, B))
    assert env.transect == (A, B)


def test_geodesic_waypoints_rejects_antipodal_endpoints():
    # Antipodal endpoints are joined by infinitely many great circles, and the
    # slerp's 1/sin(ang) returned waypoints that did not lie on the ranges it
    # reported: (0, 0) → (0, 180) put waypoint 1 at 3921 km from the start
    # while ranges_m called it 5004 km.
    from uacpy.data._geo import geodesic_waypoints
    antipodal = [((0.0, 0.0), (0.0, 180.0)),
                 ((45.0, 10.0), (-45.0, -170.0)),
                 ((10.0, 20.0), (-10.0, -160.0))]
    for start, end in antipodal:
        with pytest.raises(ConfigurationError, match='antipodal'):
            geodesic_waypoints(start, end, 5)


def test_geodesic_waypoints_hold_their_ranges_just_short_of_antipodal():
    # Outside the guard the waypoints must still sit on the ranges reported
    # for them, well inside the spherical model's own accuracy.
    from uacpy.data._geo import geodesic_waypoints, great_circle_km
    end = (0.0, 180.0 - np.degrees(1e-4))
    lats, lons, ranges_m = geodesic_waypoints((0.0, 0.0), end, 5)
    measured_m = great_circle_km(0.0, 0.0, lats, lons) * 1000.0
    assert np.allclose(measured_m, ranges_m, atol=1.0)


def test_parse_date_accepts_iso_and_objects():
    import datetime as dt
    from uacpy.data._time import parse_date
    assert parse_date('2026-06-14') == dt.date(2026, 6, 14)
    assert parse_date('2026-06-14T12:30:00') == dt.date(2026, 6, 14)
    assert parse_date(dt.date(2026, 6, 14)) == dt.date(2026, 6, 14)
    assert parse_date(dt.datetime(2026, 6, 14, 5)) == dt.date(2026, 6, 14)


@pytest.mark.parametrize('bad', ['2026-13-99', 'June 2026', '', 20260614, None])
def test_parse_date_rejects_bad(bad):
    from uacpy.data._time import parse_date
    with pytest.raises(ConfigurationError):
        parse_date(bad)
