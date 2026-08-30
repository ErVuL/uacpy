"""Tests for the on-demand bathymetry fetch (uacpy.data.bathymetry).

The unit tests stub the HTTP layer so they run fully offline; one
``requires_network``-marked test exercises the live OpenTopoData endpoint
(auto-skipped offline).
"""

import threading
import time

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import bathymetry
from uacpy.data.bathymetry import bathy_transect_plan, fetch_bathy_transect


def _ok(elevations):
    """Build an OpenTopoData-shaped OK payload from elevation values."""
    return {
        'status': 'OK',
        'results': [{'elevation': e, 'location': {'lat': 0, 'lng': 0}}
                    for e in elevations],
    }


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    """Disable the OpenTopoData ≤1 req/s spacing so chunked tests run instantly.

    The throttle itself is exercised explicitly in ``test_rate_limit_*``.
    """
    monkeypatch.setattr(bathymetry, 'OPENTOPODATA_MIN_INTERVAL_S', 0.0)


@pytest.fixture
def stub_http(monkeypatch):
    """Patch the HTTP getter; returns a list capturing requested URLs.

    The replacement reads ``elevations`` from the fixture's closure so each
    test sets the depths it expects back.
    """
    state = {'elevations': [-2458.0], 'urls': []}

    def fake_get(url, *, timeout, verbose):
        state['urls'].append(url)
        return _ok(state['elevations'])

    monkeypatch.setattr(bathymetry, '_http_get_json', fake_get)
    return state


def test_point_depth_converts_elevation_sign(stub_http):
    stub_http['elevations'] = [-2458.0]
    depth = bathymetry.fetch_bathy((43.2, 7.5))
    assert depth == pytest.approx(2458.0)
    # Result is directly usable as a scalar bathymetry.
    env = uacpy.Environment(name='pt', bathymetry=depth, ssp=1500)
    assert env.depth == pytest.approx(2458.0)


def test_longitude_normalized_to_pm180(stub_http):
    # lon given in [0, 360) must be wrapped before hitting OpenTopoData.
    stub_http['elevations'] = [-2458.0]
    bathymetry.fetch_bathy((0.0, 220.0))
    assert '-140.000000' in stub_http['urls'][-1]   # 220 - 360
    assert '220.000000' not in stub_http['urls'][-1]


def test_point_on_land_raises(stub_http):
    stub_http['elevations'] = [2359.0]
    with pytest.raises(DataFetchError, match='on land'):
        bathymetry.fetch_bathy((45.0, 6.0))


def test_null_elevation_raises(monkeypatch):
    monkeypatch.setattr(
        bathymetry, '_http_get_json',
        lambda url, *, timeout, verbose: {
            'status': 'OK',
            'results': [{'elevation': None, 'location': {'lat': 0, 'lng': 0}}],
        },
    )
    with pytest.raises(DataFetchError, match='no data'):
        bathymetry.fetch_bathy((0.0, 0.0))


def test_non_ok_status_raises(monkeypatch):
    monkeypatch.setattr(
        bathymetry, '_http_get_json',
        lambda url, *, timeout, verbose: {'status': 'INVALID', 'error': 'bad'},
    )
    with pytest.raises(DataFetchError, match='status'):
        bathymetry.fetch_bathy((0.0, 0.0))


def test_transect_shape_and_range_axis(stub_http):
    elevs = [-2458.0, -2644.0, -2692.0, -2700.0, -2710.0]
    stub_http['elevations'] = elevs
    bathy = bathymetry.fetch_bathy_transect((43.2, 7.5), (42.8, 8.1), n_points=5)

    assert bathy.shape == (5, 2)
    assert bathy[0, 0] == 0.0                      # range starts at 0
    assert np.all(np.diff(bathy[:, 0]) > 0)        # monotonic increasing range
    np.testing.assert_allclose(bathy[:, 1], [-e for e in elevs])
    # Directly consumable as range-dependent bathymetry.
    env = uacpy.Environment(name='slope', bathymetry=bathy, ssp=1500)
    assert env.has_range_dependent_bathymetry


def test_transect_range_matches_known_distance(stub_http):
    # 1 degree of latitude ≈ 111.2 km along a meridian.
    stub_http['elevations'] = [-1000.0, -1000.0]
    bathy = bathymetry.fetch_bathy_transect((0.0, 0.0), (1.0, 0.0), n_points=2)
    assert bathy[-1, 0] == pytest.approx(111_195.0, rel=1e-3)


def test_transect_requires_two_points():
    with pytest.raises(ConfigurationError, match='n_points'):
        bathymetry.fetch_bathy_transect((0.0, 0.0), (1.0, 1.0), n_points=1)


def test_transect_coincident_endpoints_raise():
    with pytest.raises(ConfigurationError, match='coincide'):
        bathymetry.fetch_bathy_transect((1.0, 2.0), (1.0, 2.0), n_points=10)


def test_transect_chunks_large_requests(monkeypatch):
    # 250 points must split into 3 calls (100 + 100 + 50).
    calls = {'n': 0}

    def fake_get(url, *, timeout, verbose):
        calls['n'] += 1
        count = url.count('%2C')  # one comma (encoded) per lat,lon pair
        return _ok([-1500.0] * count)

    monkeypatch.setattr(bathymetry, '_http_get_json', fake_get)
    bathy = bathymetry.fetch_bathy_transect((0.0, 0.0), (2.0, 2.0),
                                            n_points=250, max_points=250)
    assert bathy.shape == (250, 2)
    assert calls['n'] == 3


def test_fetch_bathy_grid(stub_http):
    # 3×3 grid; one land cell (positive elevation) becomes NaN, no raise.
    stub_http['elevations'] = [-1000, -1100, -1200,
                               -900, 50.0, -1300,
                               -800, -850, -950]
    lats, lons, depth = bathymetry.fetch_bathy_grid((40, 42), (7, 9), n_lat=3, n_lon=3)
    assert lats.shape == (3,) and lons.shape == (3,) and depth.shape == (3, 3)
    assert depth[0, 0] == 1000.0
    assert np.isnan(depth[1, 1])       # land cell → NaN (point fetchers would raise)


def test_fetch_bathy_grid_latitudes_ascend_for_descending_range(stub_http):
    # A (max, min) lat_range yields the same ascending latitude axis as
    # (min, max) — all four sources share that order.
    stub_http['elevations'] = [-1000, -1100, -1200,
                               -900, -950, -1300,
                               -800, -850, -975]
    lats, lons, depth = bathymetry.fetch_bathy_grid((42, 40), (7, 9),
                                                    n_lat=3, n_lon=3)
    assert np.all(np.diff(lats) > 0)
    assert lats[0] == 40.0 and lats[-1] == 42.0
    assert depth[0, 0] == 1000.0          # first fetched row = southernmost


def test_fetch_grid_too_small_raises():
    with pytest.raises(ConfigurationError, match='>= 2'):
        bathymetry.fetch_bathy_grid((40, 42), (7, 9), n_lat=1, n_lon=5)


def test_fetch_grid_allows_50x50(stub_http):
    # 50×50 = 2500 points (25 requests) is within budget; just check it runs.
    stub_http['elevations'] = [-1000.0] * 100   # any 100-long chunk reply
    lats, lons, depth = bathymetry.fetch_bathy_grid((36, 44), (0, 10))   # defaults 50×50
    assert depth.shape == (50, 50)


def test_fetch_grid_too_many_requests_raises():
    with pytest.raises(ConfigurationError, match='requests'):
        bathymetry.fetch_bathy_grid((0, 10), (0, 10), n_lat=110, n_lon=110)


def test_rate_limit_spaces_public_host_chunks(monkeypatch):
    # Two chunks against the public host must be spaced by the throttle.
    monkeypatch.setattr(bathymetry, 'OPENTOPODATA_MIN_INTERVAL_S', 1.0)
    monkeypatch.setattr(bathymetry, '_last_request_monotonic', 0.0)
    sleeps = []
    monkeypatch.setattr(bathymetry.time, 'sleep', lambda s: sleeps.append(s))
    monkeypatch.setattr(bathymetry, '_http_get_json',
                        lambda url, *, timeout, verbose: _ok([-1500.0] * url.count('%2C')))
    bathymetry.fetch_bathy_transect((0.0, 0.0), (2.0, 2.0),
                                    n_points=150, max_points=150)  # 2 chunks
    assert any(s > 0.0 for s in sleeps)


def test_rate_limit_skipped_for_self_hosted(monkeypatch):
    # A self-hosted base_url lifts the limit → no throttle sleep.
    monkeypatch.setattr(bathymetry, 'OPENTOPODATA_MIN_INTERVAL_S', 1.0)
    monkeypatch.setattr(bathymetry, '_last_request_monotonic', 0.0)
    sleeps = []
    monkeypatch.setattr(bathymetry.time, 'sleep', lambda s: sleeps.append(s))
    monkeypatch.setattr(bathymetry, '_http_get_json',
                        lambda url, *, timeout, verbose: _ok([-1500.0] * url.count('%2C')))
    bathymetry.fetch_bathy_transect((0.0, 0.0), (2.0, 2.0), n_points=150,
                                    max_points=150, base_url='http://localhost:5000/v1')
    assert sleeps == []


@pytest.mark.requires_network
def test_live_opentopodata_point():
    try:
        depth = bathymetry.fetch_bathy((43.2, 7.5))
    except DataFetchError as exc:
        pytest.skip(f"OpenTopoData unreachable: {exc.message}")
    # Ligurian Sea abyssal plain — a few thousand metres deep.
    assert 1000.0 < depth < 4000.0


def test_the_rate_limiter_serializes_concurrent_callers(monkeypatch):
    """Unguarded, every thread read the same stale timestamp and fired at once:
    eight calls at a 0.2 s interval left 0.2 ms between them."""
    from uacpy.data import bathymetry
    monkeypatch.setattr(bathymetry, 'OPENTOPODATA_MIN_INTERVAL_S', 0.05)
    monkeypatch.setattr(bathymetry, '_last_request_monotonic', 0.0)
    stamps, guard = [], threading.Lock()

    def worker():
        bathymetry._rate_limit()
        with guard:
            stamps.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stamps.sort()
    gaps = np.diff(stamps)
    assert gaps.min() >= 0.045, f"calls {gaps.min() * 1e3:.1f} ms apart"


class TestBathyTransectNPointsForms:
    """``n_points`` has exactly the two documented forms — an integer
    >= 2, or ``'auto'`` — and any other string raises a typed error
    naming them, before any fetch happens (previously a raw
    ``ValueError`` from ``int('lots')``)."""

    def test_a_non_auto_string_raises_naming_the_valid_forms(self):
        with pytest.raises(ConfigurationError,
                           match=r"n_points='lots' is not a sample count"):
            bathy_transect_plan((0.0, 0.0), (1.0, 0.0), n_points='lots')

    def test_fetch_bathy_transect_raises_the_same_error_before_fetching(self):
        with pytest.raises(ConfigurationError, match=r"'auto'"):
            fetch_bathy_transect((0.0, 0.0), (1.0, 0.0), n_points='lots')

    def test_none_raises_a_typed_error(self):
        with pytest.raises(ConfigurationError,
                           match="is not a sample count"):
            bathy_transect_plan((0.0, 0.0), (1.0, 0.0), n_points=None)

    def test_integer_and_auto_forms_resolve_the_plan(self):
        assert bathy_transect_plan(
            (0.0, 0.0), (1.0, 0.0), n_points=5)['n_points'] == 5
        assert bathy_transect_plan(
            (0.0, 0.0), (1.0, 0.0), n_points='auto')['n_points'] >= 2


@pytest.mark.parametrize('bad', [2.5, 'x'], ids=['non-integral', 'non-numeric'])
def test_a_bad_sample_count_raises_the_typed_configuration_error(bad):
    from uacpy.data.bathymetry import bathy_transect_plan
    with pytest.raises(ConfigurationError, match='is not a sample count'):
        bathy_transect_plan((45.0, -30.0), (46.0, -30.0), n_points=bad)
