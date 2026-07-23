"""Tests for the surface-metocean fetchers (wind, waves, sea surface).

The ERDDAP / Copernicus HTTP layers are stubbed with canned responses so these
run offline; ``requires_network`` tests hit the live services.
"""

import numpy as np
import pytest

import uacpy.data as data
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import (
    environment as env_mod, sea_surface, waves as waves_mod, wind_live,
    wind_local, ww3_live,
)

_WIND_CSV = ("time,zlev,latitude,longitude,windspeed\n"
             "UTC,m,degrees_north,degrees_east,m s-1\n"
             "2020-01-01T00:00:00Z,10.0,50.0,0.0,{v}\n")
_WW3_CSV = ("time,latitude,longitude,Thgt\n"
            "UTC,degrees_north,degrees_east,m\n"
            "2020-01-01T00:00:00Z,50.0,0.0,{v}\n")


# ── wind (live NBS) ───────────────────────────────────────────────────────────

def test_fetch_wind_parses_speed(monkeypatch):
    monkeypatch.setattr(wind_live, 'http_get',
                        lambda url, **kw: _WIND_CSV.format(v=7.3).encode())
    assert wind_live.fetch_wind((50.0, 0.0), date='2020-01-01') == pytest.approx(7.3)


def test_fetch_wind_components_fallback(monkeypatch):
    # No scalar speed var (brackets are literal in the griddap query) → the
    # fetcher combines the u, v = 3, 4 components → 5 m/s.
    def fake(url, *, timeout, verbose, source='data', user_agent='uacpy'):
        if 'windspeed[(' in url or 'wind_speed[(' in url or 'w[(' in url:
            raise DataFetchError("no such variable")
        val = 3.0 if 'u[(' in url else 4.0
        return _WIND_CSV.replace('windspeed', 'x').format(v=val).encode()
    monkeypatch.setattr(wind_live, 'http_get', fake)
    assert wind_live.fetch_wind((50.0, 0.0), date='2020-01-01') == pytest.approx(5.0)


def test_fetch_wind_bad_source_raises():
    with pytest.raises(ConfigurationError, match='wind source'):
        wind_live.fetch_wind((50.0, 0.0), date='2020-01-01', source='nope')


def test_fetch_wind_transect(monkeypatch):
    monkeypatch.setattr(wind_live, 'http_get',
                        lambda url, **kw: _WIND_CSV.format(v=6.0).encode())
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
    monkeypatch.setattr(ww3_live, 'http_get',
                        lambda url, **kw: _WW3_CSV.format(v=2.4).encode())
    assert ww3_live.fetch_hs((50.0, 0.0), date='2020-01-01') == pytest.approx(2.4)


def test_ww3_land_raises(monkeypatch):
    monkeypatch.setattr(ww3_live, 'http_get',
                        lambda url, **kw: _WW3_CSV.format(v='NaN').encode())
    with pytest.raises(DataFetchError, match='no wave height'):
        ww3_live.fetch_hs((50.0, 0.0), date='2020-01-01')


def test_fetch_waves_auto_falls_to_ww3(monkeypatch):
    # Copernicus unavailable → 'auto' falls through to WW3.
    def no_copernicus(point, **kw):
        raise DataFetchError("no login")
    monkeypatch.setattr('uacpy.data.copernicus.fetch_waves_operational',
                        no_copernicus)
    monkeypatch.setattr(ww3_live, 'http_get',
                        lambda url, **kw: _WW3_CSV.format(v=1.8).encode())
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


@pytest.mark.requires_network
def test_live_nbs_wind():
    try:
        u = wind_live.fetch_wind((45.0, -30.0), date='2020-01-15')
    except DataFetchError as exc:
        pytest.skip(f"NBS unreachable: {exc.message}")
    assert 0.0 <= u < 60.0
