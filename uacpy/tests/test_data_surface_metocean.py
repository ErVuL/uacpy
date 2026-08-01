"""Tests for the surface-metocean fetchers (wind, waves, sea surface).

The ERDDAP / Copernicus HTTP layers are stubbed with canned responses so these
run offline; ``requires_network`` tests hit the live services.
"""

import re
import urllib.parse
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import uacpy.data as data
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import (
    environment as env_mod, sea_surface, waves as waves_mod, wind_live,
    wind_local, ww3_live,
)

_WIND_CSV = ("time,zlev,latitude,longitude,{var}\n"
             "UTC,m,degrees_north,degrees_east,m s-1\n"
             "2020-01-01T00:00:00Z,10.0,50.0,330.0,{v}\n")
_WW3_CSV = ("time,depth,latitude,longitude,Thgt\n"
            "UTC,m,degrees_north,degrees_east,m\n"
            "2020-01-01T00:00:00Z,0.0,50.0,0.0,{v}\n")


def _selectors(url):
    """(variable, [axis selector values]) from a griddap query URL."""
    query = urllib.parse.unquote(url.split('?', 1)[1])
    return query.split('[', 1)[0], re.findall(r'\[\(([^)]*)\)\]', query)


def _nbs_server(values):
    """Stub of the real ``noaacwBlendedWinds6hr`` griddap: only the variables
    in ``values`` exist, 4 axes (time, zlev, latitude, longitude), longitude
    axis [0.0, 359.75]."""
    def fake(url, *, timeout=60.0, verbose=False, source='data',
             user_agent='uacpy'):
        var, sel = _selectors(url)
        if var not in values:
            raise DataFetchError(
                f"HTTP 404: Query error: variable={var} wasn't found "
                f"in datasetID=noaacwBlendedWinds6hr.")
        if len(sel) != 4:
            raise DataFetchError(
                "HTTP 400: Query error: Constraint does not match the 4 axes "
                "[time][zlev][latitude][longitude].")
        lon = float(sel[3])
        if not 0.0 <= lon < 360.0:
            raise DataFetchError(
                f"HTTP 404: Query error: longitude={lon} is outside the axis "
                f"actual_range [0.0, 359.75].")
        return _WIND_CSV.format(var=var, v=values[var]).encode()
    return fake


def _ww3_server(v):
    """Stub of the real PacIOOS ``ww3_global`` griddap: variable ``Thgt`` with
    4 axes (time, depth, latitude, longitude), longitude axis [0.0, 359.5]."""
    def fake(url, *, timeout=60.0, verbose=False, source='data',
             user_agent='uacpy'):
        var, sel = _selectors(url)
        if var != 'Thgt':
            raise DataFetchError(
                f"HTTP 404: Query error: variable={var} wasn't found in "
                f"datasetID=ww3_global.")
        if len(sel) != 4:
            raise DataFetchError(
                "HTTP 400: Query error: Constraint does not match the 4 axes "
                "[time][depth][latitude][longitude].")
        lon = float(sel[3])
        if not 0.0 <= lon < 360.0:
            raise DataFetchError(
                f"HTTP 404: Query error: longitude={lon} is outside the axis "
                f"actual_range [0.0, 359.5].")
        return _WW3_CSV.format(v=v).encode()
    return fake


# ── wind (live NBS) ───────────────────────────────────────────────────────────

def test_fetch_wind_real_nbs_schema(monkeypatch):
    # The real dataset exposes only u_wind / v_wind → √(3² + 4²) = 5 m/s.
    monkeypatch.setattr(wind_live, 'http_get',
                        _nbs_server({'u_wind': 3.0, 'v_wind': 4.0}))
    assert wind_live.fetch_wind((50.0, 0.0), date='2020-01-01') == pytest.approx(5.0)


def test_fetch_wind_western_longitude(monkeypatch):
    # lon=-30 must be sent as 330 on the [0, 360) axis.
    monkeypatch.setattr(wind_live, 'http_get',
                        _nbs_server({'u_wind': 3.0, 'v_wind': 4.0}))
    assert wind_live.fetch_wind((45.0, -30.0), date='2020-01-15') == pytest.approx(5.0)


def test_fetch_wind_scalar_speed_tolerance(monkeypatch):
    # A host exposing a scalar speed variable is still honoured first.
    monkeypatch.setattr(wind_live, 'http_get', _nbs_server({'windspeed': 7.3}))
    assert wind_live.fetch_wind((50.0, 0.0), date='2020-01-01') == pytest.approx(7.3)


def test_fetch_wind_generic_uv_fallback(monkeypatch):
    monkeypatch.setattr(wind_live, 'http_get', _nbs_server({'u': 3.0, 'v': 4.0}))
    assert wind_live.fetch_wind((50.0, 0.0), date='2020-01-01') == pytest.approx(5.0)


def test_fetch_wind_land_raises(monkeypatch):
    monkeypatch.setattr(wind_live, 'http_get',
                        _nbs_server({'u_wind': 'NaN', 'v_wind': 'NaN'}))
    with pytest.raises(DataFetchError, match='no wind'):
        wind_live.fetch_wind((50.0, 0.0), date='2020-01-01')


def test_fetch_wind_bad_source_raises():
    with pytest.raises(ConfigurationError, match='wind source'):
        wind_live.fetch_wind((50.0, 0.0), date='2020-01-01', source='nope')


def test_fetch_wind_transect(monkeypatch):
    monkeypatch.setattr(wind_live, 'http_get',
                        _nbs_server({'u_wind': 0.0, 'v_wind': 6.0}))
    ranges, speeds = wind_live.fetch_wind_transect(
        (50.0, 0.0), (50.5, 0.5), date='2020-01-01', n_points=3)
    assert speeds.tolist() == [6.0, 6.0, 6.0]
    assert ranges.shape == (3,)


# ── wind (cached climatology) ─────────────────────────────────────────────────

@pytest.fixture
def wind_cache(tmp_path, monkeypatch):
    root = tmp_path / 'wind_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    wind_local._CLIM.clear()
    wdir = root / 'wind'; wdir.mkdir(parents=True)
    lat = np.linspace(-89.5, 89.5, 12)
    lon = np.linspace(-179.5, 179.5, 24)
    speed = np.full((12, lat.size, lon.size), np.nan)
    speed[2, 6, 12] = 8.5                        # March, near (0, 0)
    np.savez_compressed(wdir / wind_local.WIND_FILE, lat=lat, lon=lon, speed=speed)
    return root


def test_wind_local_reads_climatology(wind_cache):
    # Nearest cell to (0.6, 0.6) in March is index (6, 12) = 8.5 m/s.
    assert wind_local.wind_speed((0.6, 0.6), date='2021-03-15') == pytest.approx(8.5)


def test_wind_local_land_raises(wind_cache):
    with pytest.raises(DataFetchError, match='land'):
        wind_local.wind_speed((0.6, 0.6), date='2021-07-15')   # July = NaN


def test_wind_local_missing_cache_names_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    wind_local._CLIM.clear()
    with pytest.raises(ConfigurationError, match='install.sh --data wind'):
        wind_local.wind_speed((0.6, 0.6), date='2021-03-15')


def test_fetch_wind_local_dispatch(wind_cache):
    assert data.fetch_wind((0.6, 0.6), date='2021-03-15',
                           source='local') == pytest.approx(8.5)


# ── waves ─────────────────────────────────────────────────────────────────────

def test_ww3_fetch_hs(monkeypatch):
    monkeypatch.setattr(ww3_live, 'http_get', _ww3_server(2.4))
    assert ww3_live.fetch_hs((50.0, 0.0), date='2020-01-01') == pytest.approx(2.4)


def test_ww3_western_longitude(monkeypatch):
    # lon=-158 must be sent as 202 on the [0, 360) axis.
    monkeypatch.setattr(ww3_live, 'http_get', _ww3_server(2.4))
    assert ww3_live.fetch_hs((21.0, -158.0), date='2020-01-01') == pytest.approx(2.4)


def test_ww3_land_raises(monkeypatch):
    monkeypatch.setattr(ww3_live, 'http_get', _ww3_server('NaN'))
    with pytest.raises(DataFetchError, match='no wave height'):
        ww3_live.fetch_hs((50.0, 0.0), date='2020-01-01')


def test_fetch_waves_auto_falls_to_ww3(monkeypatch):
    # Copernicus unavailable → 'auto' falls through to WW3.
    def no_copernicus(point, **kw):
        raise DataFetchError("no login")
    monkeypatch.setattr('uacpy.data.copernicus.fetch_waves_operational',
                        no_copernicus)
    monkeypatch.setattr(ww3_live, 'http_get', _ww3_server(1.8))
    out = waves_mod.fetch_waves((50.0, 0.0), date='2020-01-01')
    assert out['hs'] == pytest.approx(1.8) and out['source'] == 'ww3'


def test_fetch_waves_bad_source_raises():
    with pytest.raises(ConfigurationError, match='wave source'):
        waves_mod.fetch_waves((50.0, 0.0), date='2020-01-01', source='nope')


# ── sea surface ───────────────────────────────────────────────────────────────

def test_hs_to_pm_wind():
    # Hs = 0.021·U² → U = sqrt(Hs/0.021); Hs=2.1 → ~10 m/s.
    assert sea_surface.hs_to_pm_wind(2.1) == pytest.approx(10.0, rel=1e-3)


def test_fetch_sea_surface_from_waves(monkeypatch):
    monkeypatch.setattr(waves_mod, 'fetch_waves',
                        lambda point, **kw: {'hs': 2.1, 'tp': 8.0, 'source': 'waverys'})
    alt, src = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', max_range=5000.0, n_points=64, seed=1)
    assert src == 'waverys'
    assert alt.shape == (64, 2)
    assert np.all(np.isfinite(alt))
    assert alt[0, 0] == 0.0 and alt[-1, 0] == pytest.approx(5000.0)


def test_fetch_sea_surface_wind_fallback(monkeypatch):
    monkeypatch.setattr(waves_mod, 'fetch_waves',
                        lambda point, **kw: (_ for _ in ()).throw(DataFetchError("no waves")))
    monkeypatch.setattr(wind_live, 'fetch_wind', lambda point, **kw: 10.0)
    alt, src = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', max_range=5000.0, n_points=32, seed=2)
    assert src == 'nbs' and alt.shape == (32, 2)


def test_fetch_sea_surface_local_reads_the_cached_climatology(wind_cache,
                                                              monkeypatch):
    """source='local' must reach the installed wind grid without the network."""
    def boom(url, **kw):
        raise AssertionError(f"network call in a local sea-state fetch: {url}")

    monkeypatch.setattr(wind_live, 'http_get', boom)
    alt, src = sea_surface.fetch_sea_surface(
        (0.6, 0.6), date='2021-03-15', max_range=5000.0, n_points=32, seed=3,
        source='local')
    assert src == 'nbs' and alt.shape == (32, 2)


def test_fetch_sea_surface_auto_falls_back_to_the_climatology(wind_cache,
                                                              monkeypatch):
    """'auto' ends on the cached climatology when waves and live wind fail."""
    monkeypatch.setattr(waves_mod, 'fetch_waves',
                        lambda point, **kw: (_ for _ in ()).throw(DataFetchError("no waves")))
    monkeypatch.setattr(
        wind_live, '_wind_speed',
        lambda *a, **kw: (_ for _ in ()).throw(DataFetchError("erddap down")))
    alt, src = sea_surface.fetch_sea_surface(
        (0.6, 0.6), date='2021-03-15', max_range=5000.0, n_points=32, seed=4,
        source='auto')
    assert src == 'nbs' and alt.shape == (32, 2)


def test_fetch_environment_altimetry_local(wind_cache, monkeypatch):
    """The installed wind climatology is reachable through fetch_environment."""
    def boom(url, **kw):
        raise AssertionError(f"network call in a local sea-state fetch: {url}")

    monkeypatch.setattr(wind_live, 'http_get', boom)
    env = env_mod.fetch_environment(
        (0.6, 0.6), bathymetry=2000.0, ssp=1500.0, date='2021-03-15',
        transect_to=(0.9, 0.9), altimetry_sources='local',
        sea_surface_n_points=32, sea_surface_seed=5)
    assert env.altimetry is not None
    assert 'nbs' in [s.source.id for s in env.data_sources]


# ── fetch_environment altimetry integration ───────────────────────────────────

def test_altimetry_requires_transect():
    with pytest.raises(ConfigurationError, match='requires transect_to'):
        env_mod.fetch_environment((50.0, 0.0), bathymetry=2000.0, ssp=1500.0,
                                  date='2020-01-01', altimetry_sources='waves')


def test_altimetry_requires_date():
    with pytest.raises(ConfigurationError, match='needs date'):
        env_mod.fetch_environment((50.0, 0.0), bathymetry=2000.0, ssp=1500.0,
                                  transect_to=(50.5, 0.5), altimetry_sources='waves')


def test_fetch_environment_altimetry(monkeypatch):
    # Literal bathy/ssp keep it offline; the sea-surface fetch is stubbed and its
    # provenance id must land in env.data_sources.
    alt = np.column_stack([np.linspace(0, 5000, 10), np.zeros(10)])
    monkeypatch.setattr(sea_surface, 'fetch_sea_surface',
                        lambda point, **kw: (alt, 'waverys'))
    env = env_mod.fetch_environment(
        (50.0, 0.0), bathymetry=2000.0, ssp=1500.0, transect_to=(50.5, 0.5),
        date='2020-01-01', altimetry_sources='waves')
    assert env.altimetry is not None
    assert 'waverys' in [s.source.id for s in env.data_sources]


_NETWORK_DOWN_TOKENS = ('could not reach', 'timed out', 'timeout', 'connection',
                        'unreachable', 'http 502', 'http 503', 'http 504')


def _skip_or_fail(exc, service):
    """Skip only on network-level failure; a structured server rejection
    (bad variable / constraint / axis) is a real bug and must fail."""
    if any(tok in exc.message.lower() for tok in _NETWORK_DOWN_TOKENS):
        pytest.skip(f"{service} unreachable: {exc.message}")
    pytest.fail(f"{service} rejected the query: {exc.message}")


@pytest.mark.requires_network
def test_live_nbs_wind():
    try:
        u = wind_live.fetch_wind((45.0, -30.0), date='2020-01-15')
    except DataFetchError as exc:
        _skip_or_fail(exc, 'NBS')
    assert 0.0 <= u < 60.0


@pytest.mark.requires_network
def test_live_ww3_hs():
    # Operational feed = rolling recent window; western-hemisphere point.
    when = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        hs = ww3_live.fetch_hs((21.0, -158.0), date=when)
    except DataFetchError as exc:
        _skip_or_fail(exc, 'WaveWatch III')
    assert 0.0 <= hs < 30.0
