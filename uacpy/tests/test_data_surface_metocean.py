"""Tests for the surface-metocean fetchers (wind, waves, sea surface).

The ERDDAP / Copernicus HTTP layers are stubbed with canned responses so these
run offline; ``requires_network`` tests hit the live services.
"""

import pathlib
import re
import urllib.parse
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

# fetch_wind answers in knots; the fake servers below speak NBS's m/s
from uacpy.core.units import ms_to_knots

import uacpy.data as data
from uacpy.data import _http
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import (
    environment as env_mod, sea_surface, waves as waves_mod, wind_live,
    wind_local, ww3_live,
)
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.tests._cache_builders import _skip_or_fail

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
    assert wind_live.fetch_wind((50.0, 0.0), date='2020-01-01') == pytest.approx(ms_to_knots(5.0))


def test_fetch_wind_western_longitude(monkeypatch):
    # lon=-30 must be sent as 330 on the [0, 360) axis.
    monkeypatch.setattr(wind_live, 'http_get',
                        _nbs_server({'u_wind': 3.0, 'v_wind': 4.0}))
    assert wind_live.fetch_wind((45.0, -30.0), date='2020-01-15') == pytest.approx(ms_to_knots(5.0))


def test_fetch_wind_scalar_speed_tolerance(monkeypatch):
    # A host exposing a scalar speed variable is still honoured first.
    monkeypatch.setattr(wind_live, 'http_get', _nbs_server({'windspeed': 7.3}))
    assert wind_live.fetch_wind((50.0, 0.0), date='2020-01-01') == pytest.approx(ms_to_knots(7.3))


def test_fetch_wind_generic_uv_fallback(monkeypatch):
    monkeypatch.setattr(wind_live, 'http_get', _nbs_server({'u': 3.0, 'v': 4.0}))
    assert wind_live.fetch_wind((50.0, 0.0), date='2020-01-01') == pytest.approx(ms_to_knots(5.0))


def test_fetch_wind_land_raises(monkeypatch):
    monkeypatch.setattr(wind_live, 'http_get',
                        _nbs_server({'u_wind': 'NaN', 'v_wind': 'NaN'}))
    with pytest.raises(DataFetchError, match='no wind'):
        wind_live.fetch_wind((50.0, 0.0), date='2020-01-01')


def test_fetch_wind_bad_source_raises():
    with pytest.raises(ConfigurationError, match='wind source'):
        wind_live.fetch_wind((50.0, 0.0), date='2020-01-01', source='nope')


def test_fetch_wind_names_the_live_source_by_its_catalogue_id(monkeypatch):
    """The id a provenance record carries is the id the fetcher takes; the
    transport ('erddap') is not a source name."""
    from uacpy.data.sources import SOURCES
    monkeypatch.setattr(wind_live, 'http_get',
                        _nbs_server({'u_wind': 3.0, 'v_wind': 4.0}))
    assert 'nbs' in SOURCES
    assert wind_live.fetch_wind((50.0, 0.0), date='2020-01-01',
                                source='nbs') == pytest.approx(ms_to_knots(5.0))
    with pytest.raises(ConfigurationError, match='wind source'):
        wind_live.fetch_wind((50.0, 0.0), date='2020-01-01', source='erddap')


def test_fetch_wind_transect(monkeypatch):
    monkeypatch.setattr(wind_live, 'http_get',
                        _nbs_server({'u_wind': 0.0, 'v_wind': 6.0}))
    track = wind_live.fetch_wind_transect(
        (50.0, 0.0), (50.5, 0.5), date='2020-01-01', n_points=3)
    assert track.data == pytest.approx([ms_to_knots(6.0)] * 3)
    assert track.ranges.shape == (3,)
    assert (track.lats[0], track.lons[0]) == (50.0, 0.0)
    assert track.provenance.requested_date == '2020-01-01'
    pytest.importorskip('xarray')
    ds = track.to_xarray()
    assert ds['data'].dims == ('range',)
    assert ds['data'].attrs['units'] == 'kn'
    assert ds['lat'].dims == ('range',)
    back = type(track).from_xarray(ds)
    assert back.data == pytest.approx([ms_to_knots(6.0)] * 3)
    assert back.quantity == 'wind_speed'


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
                           source='local') == pytest.approx(ms_to_knots(8.5))


# ── waves ─────────────────────────────────────────────────────────────────────

def test_ww3_fetch_hs(monkeypatch):
    monkeypatch.setattr(ww3_live, 'http_get', _ww3_server(2.4))
    assert ww3_live.fetch_hs((50.0, 0.0), date='2020-01-01') == pytest.approx(2.4)


@pytest.mark.parametrize('value, answers', [(0.0, True), (-999.0, False)],
                         ids=['calm', 'negative-fill'])
def test_a_negative_wave_height_reads_as_no_value(monkeypatch, value, answers):
    monkeypatch.setattr(ww3_live, 'http_get', _ww3_server(value))
    if answers:
        assert ww3_live.fetch_hs((50.0, 0.0), date='2020-01-01') == value
    else:
        with pytest.raises(DataFetchError, match='no wave height'):
            ww3_live.fetch_hs((50.0, 0.0), date='2020-01-01')


@pytest.mark.parametrize('value, answers', [(0.0, True), (-999.0, False)],
                         ids=['calm', 'negative-fill'])
def test_a_negative_scalar_wind_speed_reads_as_no_value(monkeypatch, value,
                                                        answers):
    monkeypatch.setattr(wind_live, 'http_get', _nbs_server({'windspeed': value}))
    if answers:
        assert wind_live.fetch_wind((50.0, 0.0), date='2020-01-01') == value
    else:
        with pytest.raises(DataFetchError, match='no wind speed'):
            wind_live.fetch_wind((50.0, 0.0), date='2020-01-01')


def test_ww3_western_longitude(monkeypatch):
    # lon=-158 must be sent as 202 on the [0, 360) axis.
    monkeypatch.setattr(ww3_live, 'http_get', _ww3_server(2.4))
    assert ww3_live.fetch_hs((21.0, -158.0), date='2020-01-01') == pytest.approx(2.4)


def test_ww3_land_raises(monkeypatch):
    monkeypatch.setattr(ww3_live, 'http_get', _ww3_server('NaN'))
    with pytest.raises(DataFetchError, match='no wave height'):
        ww3_live.fetch_hs((50.0, 0.0), date='2020-01-01')


def test_ww3_network_failure_is_named_not_called_land(monkeypatch):
    def down(url, **kw):
        raise DataFetchError("Could not reach the host: timed out")
    monkeypatch.setattr(ww3_live, 'http_get', down)
    with pytest.raises(DataFetchError, match='timed out'):
        ww3_live.fetch_hs((50.0, 0.0), date='2020-01-01')


def test_fetch_waves_auto_falls_to_ww3(monkeypatch):
    # Copernicus unavailable → 'auto' falls through to WW3.
    def no_copernicus(point, **kw):
        raise DataFetchError("no login")
    monkeypatch.setattr('uacpy.data.copernicus.fetch_waves_operational',
                        no_copernicus)
    monkeypatch.setattr(ww3_live, 'http_get', _ww3_server(1.8))
    out = waves_mod.fetch_waves((50.0, 0.0), date='2020-01-01')
    assert type(out) is waves_mod.SeaStateRecord
    assert out.hs == pytest.approx(1.8) and out.tp is None
    assert out.provenance.source.id == 'ww3'
    assert out.provenance.requested_date == '2020-01-01'


def test_the_sea_state_saves_and_loads_with_its_provenance(tmp_path):
    record = _waves_record(1.2, source='ww3')
    path = tmp_path / 'sea.npz'
    np.savez(path, **record.to_dict())
    back = waves_mod.SeaStateRecord.from_dict(dict(np.load(path,
                                                           allow_pickle=True)))
    assert float(back.hs) == 1.2 and float(back.tp) == 8.0
    assert back.provenance.source is record.provenance.source
    assert back.provenance.requested_date == '2020-01-01'


def test_fetch_waves_bad_source_raises():
    with pytest.raises(ConfigurationError, match='wave source'):
        waves_mod.fetch_waves((50.0, 0.0), date='2020-01-01', source='nope')


def test_the_wave_source_tokens_are_the_catalogue_ids(monkeypatch):
    from uacpy.data import SOURCES
    assert waves_mod.WAVE_SOURCES == ('waverys', 'ww3')
    assert all(name in SOURCES for name in waves_mod.WAVE_SOURCES)
    record = _waves_record(1.2)
    monkeypatch.setattr('uacpy.data.copernicus.fetch_waves_operational',
                        lambda point, **kw: record)
    out = waves_mod.fetch_waves((50.0, 0.0), date='2020-01-01',
                                source='waverys')
    assert out.provenance is record.provenance
    with pytest.raises(ConfigurationError, match="'copernicus'"):
        waves_mod.fetch_waves((50.0, 0.0), date='2020-01-01',
                              source='copernicus')


def _waves_record(hs, source='waverys'):
    """What ``fetch_waves`` returns: the height and its provenance record."""
    from uacpy.data import SOURCES, DataProvenance
    return waves_mod.SeaStateRecord(
        hs=hs, tp=8.0, provenance=DataProvenance(source=SOURCES[source],
                                                 requested_point=(50.0, 0.0),
                                                 requested_date='2020-01-01'))


# ── sea surface ───────────────────────────────────────────────────────────────

def test_hs_to_pm_wind():
    # Inverts the Pierson-Moskowitz Hs = coeff·U², so it must round-trip
    # against whatever coefficient the module carries.
    for u in (5.0, 10.0, 18.0):
        hs = sea_surface._PM_HS_COEFF * u ** 2
        assert sea_surface.hs_to_pm_wind(hs) == pytest.approx(u)
    assert sea_surface.hs_to_pm_wind(-1.0) == 0.0        # clamped, not NaN


def test_fetch_sea_surface_from_waves(monkeypatch):
    monkeypatch.setattr(waves_mod, 'fetch_waves',
                        lambda point, **kw: _waves_record(2.1))
    alt = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', rmax_m=5000.0, n_points=64,
        rng=np.random.default_rng(1))
    assert alt.data_sources[0].source.id == 'waverys'
    assert alt.ranges.shape == alt.heights.shape == (64,)
    assert np.all(np.isfinite(alt.heights))
    assert alt.ranges[0] == 0.0 and alt.ranges[-1] == pytest.approx(5000.0)


def test_fetch_sea_surface_wind_fallback(monkeypatch):
    monkeypatch.setattr(waves_mod, 'fetch_waves',
                        lambda point, **kw: (_ for _ in ()).throw(DataFetchError("no waves")))
    monkeypatch.setattr(wind_live, 'wind_at', lambda point, **kw: (
        10.0, DataProvenance(source=SOURCES['nbs'],
                             requested_point=tuple(point))))
    alt = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', rmax_m=5000.0, n_points=32,
        rng=np.random.default_rng(2))
    assert alt.data_sources[0].source.id == 'nbs' and alt.ranges.size == 32


def test_a_sea_surface_draws_from_the_generator_it_is_given(monkeypatch):
    """``rng=`` like every other drawing function: equal generators give
    equal surfaces, and the generator is the one drawn from."""
    monkeypatch.setattr(waves_mod, 'fetch_waves',
                        lambda point, **kw: _waves_record(2.1))
    kw = dict(date='2020-01-01', rmax_m=5000.0, n_points=64)
    a = sea_surface.fetch_sea_surface((50.0, 0.0), rng=np.random.default_rng(9),
                                      **kw)
    b = sea_surface.fetch_sea_surface((50.0, 0.0), rng=np.random.default_rng(9),
                                      **kw)
    np.testing.assert_array_equal(a.heights, b.heights)
    shared = np.random.default_rng(9)
    first = sea_surface.fetch_sea_surface((50.0, 0.0), rng=shared, **kw)
    second = sea_surface.fetch_sea_surface((50.0, 0.0), rng=shared, **kw)
    np.testing.assert_array_equal(first.heights, a.heights)
    assert not np.array_equal(second.heights, first.heights)
    import inspect
    assert 'seed' not in inspect.signature(
        sea_surface.fetch_sea_surface).parameters


def test_fetch_sea_surface_returns_an_altimetry_carrying_its_provenance(
        monkeypatch):
    from uacpy.core.altimetry import Altimetry
    from uacpy.data import DataProvenance
    monkeypatch.setattr(waves_mod, 'fetch_waves',
                        lambda point, **kw: _waves_record(2.0))
    alt = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', rmax_m=5000.0, n_points=32,
        rng=np.random.default_rng(1))
    assert isinstance(alt, Altimetry)
    (prov,) = alt.data_sources
    assert isinstance(prov, DataProvenance)
    assert prov.source.id == 'waverys'
    assert prov.requested_date == '2020-01-01'
    assert alt.ranges[-1] == pytest.approx(5000.0)
    with pytest.raises(TypeError, match='max_range'):
        sea_surface.fetch_sea_surface(
            (50.0, 0.0), date='2020-01-01', max_range=5000.0)


def test_fetch_sea_surface_local_reads_the_cached_climatology(wind_cache,
                                                              monkeypatch):
    """source='local' must reach the installed wind grid without the network."""
    def boom(url, **kw):
        raise AssertionError(f"network call in a local sea-state fetch: {url}")

    monkeypatch.setattr(wind_live, 'http_get', boom)
    alt = sea_surface.fetch_sea_surface(
        (0.6, 0.6), date='2021-03-15', rmax_m=5000.0, n_points=32,
        rng=np.random.default_rng(3),
        source='local')
    assert alt.data_sources[0].source.id == 'nbs' and alt.ranges.size == 32


def test_fetch_sea_surface_auto_falls_back_to_the_climatology(wind_cache,
                                                              monkeypatch):
    """'auto' ends on the cached climatology when waves and live wind fail."""
    monkeypatch.setattr(waves_mod, 'fetch_waves',
                        lambda point, **kw: (_ for _ in ()).throw(DataFetchError("no waves")))
    monkeypatch.setattr(
        wind_live, '_wind_speed',
        lambda *a, **kw: (_ for _ in ()).throw(DataFetchError("erddap down")))
    alt = sea_surface.fetch_sea_surface(
        (0.6, 0.6), date='2021-03-15', rmax_m=5000.0, n_points=32,
        rng=np.random.default_rng(4),
        source='auto')
    assert alt.data_sources[0].source.id == 'nbs' and alt.ranges.size == 32


def test_fetch_environment_altimetry_local(wind_cache, monkeypatch):
    """The installed wind climatology is reachable through fetch_environment."""
    def boom(url, **kw):
        raise AssertionError(f"network call in a local sea-state fetch: {url}")

    monkeypatch.setattr(wind_live, 'http_get', boom)
    env = env_mod.fetch_environment(
        (0.6, 0.6), bathymetry=2000.0, ssp=1500.0, date='2021-03-15',
        transect_to=(0.9, 0.9), altimetry_sources='local',
        altimetry_n_points=32, altimetry_rng=np.random.default_rng(5))
    assert [s.source.id for s in env.altimetry.data_sources] == ['nbs']
    (nbs,) = [s for s in env.data_sources if s.source.id == 'nbs']
    assert nbs.data_date == 'month 03 (climatology)'      # cache records no period
    assert nbs.requested_date == '2021-03-15'


def test_live_and_climatology_wind_provenance_differ(dated_wind_cache,
                                                     monkeypatch):
    monkeypatch.setattr(waves_mod, 'fetch_waves',
                        lambda point, **kw: (_ for _ in ()).throw(DataFetchError("no waves")))
    monkeypatch.setattr(wind_live, '_wind_speed', lambda *a, **kw: (7.0, None))
    kw = dict(date='2021-03-15', rmax_m=5000.0, n_points=32,
              rng=np.random.default_rng(4))
    (live,) = sea_surface.fetch_sea_surface(
        (0.6, 0.6), source='nbs', **kw).data_sources
    (clim,) = sea_surface.fetch_sea_surface(
        (0.6, 0.6), source='local', **kw).data_sources
    assert live.source.id == clim.source.id == 'nbs'
    assert live.data_date is None and live.requested_date == '2021-03-15'
    assert clim.data_date == 'month 03, 2013-2022 (climatology)'


# ── fetch_environment altimetry integration ───────────────────────────────────

def test_altimetry_requires_transect():
    with pytest.raises(ConfigurationError, match='requires transect_to'):
        env_mod.fetch_environment((50.0, 0.0), bathymetry=2000.0, ssp=1500.0,
                                  date='2020-01-01', altimetry_sources=('waverys', 'ww3'))


def test_altimetry_requires_date():
    with pytest.raises(ConfigurationError, match='needs date'):
        env_mod.fetch_environment((50.0, 0.0), bathymetry=2000.0, ssp=1500.0,
                                  transect_to=(50.5, 0.5), altimetry_sources=('waverys', 'ww3'))


def test_altimetry_guard_falls_back_to_literal():
    # altimetry= is the documented fallback when the sea-state fetch cannot
    # run; a missing date= must reach that fallback, not raise past it.
    alt = np.column_stack([np.linspace(0.0, 5000.0, 10), np.zeros(10)])
    env = env_mod.fetch_environment(
        (50.0, 0.0), bathymetry=2000.0, ssp=1500.0, transect_to=(50.5, 0.5),
        altimetry_sources=('waverys', 'ww3'), altimetry=alt)
    assert env.altimetry is not None


def test_fetch_environment_altimetry(monkeypatch):
    # Literal bathy/ssp keep it offline; the sea-surface fetch is stubbed and its
    # provenance id must land in env.data_sources.
    alt = sea_surface._altimetry(
        np.column_stack([np.linspace(0, 5000, 10), np.zeros(10)]),
        sea_surface._provenance('waverys', 50.0, 0.0, '2020-01-01'))
    monkeypatch.setattr(sea_surface, 'fetch_sea_surface',
                        lambda point, **kw: alt)
    env = env_mod.fetch_environment(
        (50.0, 0.0), bathymetry=2000.0, ssp=1500.0, transect_to=(50.5, 0.5),
        date='2020-01-01', altimetry_sources=('waverys', 'ww3'))
    assert env.altimetry is not alt             # the environment's own copy
    assert np.array_equal(env.altimetry.heights, alt.heights)
    assert env.altimetry.data_sources == alt.data_sources
    assert 'waverys' in [s.source.id for s in env.data_sources]


@pytest.mark.requires_network
def test_live_nbs_wind():
    try:
        u = wind_live.fetch_wind((45.0, -30.0), date='2020-01-15')
    except DataFetchError as exc:
        _skip_or_fail(exc, 'NBS')
    assert 0.0 <= u < 120.0                  # knots


@pytest.mark.requires_network
def test_live_ww3_hs():
    # Operational feed = rolling recent window; western-hemisphere point.
    when = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        hs = ww3_live.fetch_hs((21.0, -158.0), date=when)
    except DataFetchError as exc:
        _skip_or_fail(exc, 'WaveWatch III')
    assert 0.0 <= hs < 30.0


def _stub_waves(monkeypatch, hs):
    monkeypatch.setattr(waves_mod, 'fetch_waves',
                        lambda point, **kw: _waves_record(hs))


@pytest.mark.parametrize('rmax_m', [1e3, 1e4, 5e4, 1e5, 4.27e5])
def test_sea_surface_holds_its_wave_height_at_every_transect_length(
        rmax_m, monkeypatch):
    """The realization is sized from the sea state, so its significant wave
    height tracks the fetched one however long the transect is.

    With a fixed 500 samples the range step outgrows the Pierson-Moskowitz
    peak wavelength (64 m at U = 10 m/s) and the whole spectrum falls above
    Nyquist: the realized Hs was 86 % of the requested one over 10 km, 2.7 %
    over 50 km and numerically zero over 427 km — a silently flat sea.
    """
    hs = 2.1
    _stub_waves(monkeypatch, hs)
    alt = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', rmax_m=rmax_m,
        rng=np.random.default_rng(7))
    assert alt.data_sources[0].source.id == 'waverys'
    assert alt.ranges[0] == 0.0 and alt.ranges[-1] == pytest.approx(rmax_m)
    # Hs = 4·rms for a Gaussian sea surface.
    assert 4.0 * np.std(alt.heights) == pytest.approx(hs, rel=0.1)
    # The step resolves the peak wavelength lambda_p = 2*pi*U^2/g.
    u = sea_surface.hs_to_pm_wind(hs)
    dx = rmax_m / (alt.ranges.size - 1)
    from uacpy.core.altimetry import SEA_SURFACE_SAMPLES_PER_PEAK
    from uacpy.core.constants import STANDARD_GRAVITY_M_S2
    assert dx <= (2 * np.pi * u ** 2 / STANDARD_GRAVITY_M_S2
                  / SEA_SURFACE_SAMPLES_PER_PEAK)


def test_sea_surface_keeps_the_historical_floor_on_a_short_transect(monkeypatch):
    # 1 km at U = 10 m/s needs only ~130 samples to resolve the peak; the
    # realization keeps the old fixed default as a floor so a short transect
    # does not come back coarser than it used to.
    _stub_waves(monkeypatch, 2.1)
    alt = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', rmax_m=1000.0,
        rng=np.random.default_rng(1))
    from uacpy.core.altimetry import SEA_SURFACE_MIN_POINTS
    assert alt.ranges.size == SEA_SURFACE_MIN_POINTS


def test_pinned_n_points_too_coarse_to_resolve_the_peak_warns(monkeypatch):
    """A caller-pinned count that aliases the spectrum away warns, naming the
    step it produces and the count that would resolve the peak, instead of
    returning a flat surface silently."""
    _stub_waves(monkeypatch, 2.1)
    with pytest.warns(UserWarning, match=r'dx = 100\.2 m.*n_points >= 6\d{3}'):
        alt = sea_surface.fetch_sea_surface(
            (50.0, 0.0), date='2020-01-01', rmax_m=50000.0, n_points=500,
            rng=np.random.default_rng(1))
    assert alt.ranges.size == 500             # the pinned count is still honoured


def test_sea_surface_sizing_is_capped_and_warned_for_a_calm_long_transect(
        monkeypatch):
    # A near-calm sea has a short peak wavelength, so resolving it over a long
    # transect would ask for millions of samples: the count is capped and the
    # shortfall reported rather than allocated.
    _stub_waves(monkeypatch, 0.09)                       # U ~ 2 m/s
    with pytest.warns(UserWarning, match='cap'):
        alt = sea_surface.fetch_sea_surface(
            (50.0, 0.0), date='2020-01-01', rmax_m=4.27e5,
            rng=np.random.default_rng(1))
    from uacpy.core.altimetry import SEA_SURFACE_MAX_POINTS
    assert alt.ranges.size == SEA_SURFACE_MAX_POINTS


def test_empty_wave_source_raises_a_typed_error():
    with pytest.raises(ConfigurationError, match='No data source was tried'):
        waves_mod.fetch_waves((50.0, 0.0), date='2020-01-01', source=())


class _RecordingDataset:
    """netCDF stand-in with no data variable, so the reader raises mid-read."""

    def __init__(self):
        self.closed = False
        self.variables = {'latitude': np.array([0.5, 1.5]),
                          'longitude': np.array([0.5, 1.5]),
                          'lat': np.array([0.5, 1.5]),
                          'lon': np.array([0.5, 1.5])}

    def close(self):
        self.closed = True


@pytest.fixture
def dated_wind_cache(tmp_path, monkeypatch):
    """A wind cache that records its reference period."""
    root = tmp_path / 'dated_wind_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    wind_local._CLIM.clear()
    wdir = root / 'wind'
    wdir.mkdir(parents=True)
    lat = np.linspace(-89.5, 89.5, 12)
    lon = np.linspace(-179.5, 179.5, 24)
    speed = np.full((12, lat.size, lon.size), 8.5)
    np.savez_compressed(wdir / wind_local.WIND_FILE, lat=lat, lon=lon,
                        speed=speed,
                        years=np.arange(2013, 2023, dtype=np.int32))
    return root


def test_the_wind_cache_reports_the_period_it_was_built_over(dated_wind_cache):
    """The cache records the years it averaged, and reads them back."""
    assert wind_local.climatology_period() == '2013-2022 (climatology)'


def test_a_wind_cache_without_a_period_loads_its_grid(wind_cache):
    """Old caches and the synthetic ones the tests build carry no ``years``.
    Absent is not an error: the grid still reads, the vintage is unstated."""
    assert wind_local.climatology_period() is None
    assert wind_local.wind_speed((0.6, 0.6), date='2021-03-15') == pytest.approx(8.5)


def test_the_environment_provenance_carries_the_wind_vintage(dated_wind_cache,
                                                             monkeypatch):
    """The cached-climatology altimetry reaches ``env.data_sources`` dated by
    its month and the cache's recorded period."""
    monkeypatch.setattr(wind_live, 'http_get', lambda url, **kw: pytest.fail(
        f"network call in a local sea-state fetch: {url}"))
    env = env_mod.fetch_environment(
        (0.6, 0.6), bathymetry=2000.0, ssp=1500.0, date='2021-03-15',
        transect_to=(0.9, 0.9), altimetry_sources='local',
        altimetry_n_points=32, altimetry_rng=np.random.default_rng(5))
    (nbs,) = [s for s in env.data_sources if s.source.id == 'nbs']
    assert nbs.data_date == 'month 03, 2013-2022 (climatology)'
    from uacpy.data.environment import _climatology_vintage
    assert _climatology_vintage('gebco') is None
    assert _climatology_vintage('woa23') is None


def _published_climatology_bytes(nlat=4, nlon=8, nmonth=12):
    """A file shaped like NCEI's published climatology, in memory."""
    import netCDF4

    ds = netCDF4.Dataset('clim.nc', 'w', diskless=True, persist=False,
                         memory=1 << 20)
    ds.createDimension('month', nmonth)
    ds.createDimension('zlev', 1)
    ds.createDimension('lat', nlat)
    ds.createDimension('lon', nlon)
    ds.createVariable('lat', 'f4', ('lat',))[:] = np.linspace(-80, 80, nlat)
    ds.createVariable('lon', 'f4', ('lon',))[:] = np.linspace(0, 315, nlon)
    speed = ds.createVariable('windspeed', 'f4',
                              ('month', 'zlev', 'lat', 'lon'))
    speed[:] = np.arange(nmonth)[:, None, None, None] * np.ones(
        (1, 1, nlat, nlon))
    ds.createVariable('u_wind', 'f4', ('month', 'zlev', 'lat', 'lon'))[:] = 1.0
    return ds.close()


def test_the_climatology_is_one_request_for_the_published_file(tmp_path,
                                                               monkeypatch):
    """One request for a file NOAA has already averaged: the cache is filled
    from :data:`NBS_CLIMATOLOGY_URL` and from nothing else."""
    blob = _published_climatology_bytes()
    asked = []

    def fake_curl(url, out, *, timeout, verbose):
        asked.append(url)
        pathlib.Path(out).write_bytes(blob)
        return True

    monkeypatch.setattr(_http, 'curl_download', fake_curl)
    out = wind_local.download_wind_db(cache_dir=str(tmp_path / 'c'),
                                      verbose=False)
    assert asked == [wind_local.NBS_CLIMATOLOGY_URL]
    with np.load(out) as cached:
        assert cached['speed'].shape == (12, 4, 8), "the zlev axis survived"
        assert cached['years'].tolist() == list(wind_local.NBS_CLIMATOLOGY_YEARS)
    assert not (tmp_path / 'c' / wind_local._NBS_RAW_FILE).exists(), (
        "the 237 MB download was kept beside the 28 MB it is kept for")


def test_the_climatology_downloads_through_urllib_without_curl(tmp_path,
                                                               monkeypatch):
    """A host without curl (``curl_download`` -> False) still builds the cache."""
    blob = _published_climatology_bytes()
    asked = []

    def fake_get(url, **kw):
        asked.append(url)
        return blob

    monkeypatch.setattr(_http, 'curl_download', lambda *a, **kw: False)
    monkeypatch.setattr(_http, 'http_get', fake_get)
    out = wind_local.download_wind_db(cache_dir=str(tmp_path / 'c'),
                                      verbose=False)
    assert asked == [wind_local.NBS_CLIMATOLOGY_URL]
    with np.load(out) as cached:
        assert cached['speed'].shape == (12, 4, 8)
    assert not (tmp_path / 'c' / wind_local._NBS_RAW_FILE).exists()


def test_the_published_climatology_reports_its_own_reference_period(
        tmp_path, monkeypatch):
    """``climatology_period`` reads the cache rather than assuming a default,
    so the WMO period the published file covers is what a result records."""
    blob = _published_climatology_bytes()
    monkeypatch.setattr(
        _http, 'curl_download',
        lambda url, out, **kw: (pathlib.Path(out).write_bytes(blob), True)[1])
    cache = tmp_path / 'c'
    wind_local.download_wind_db(cache_dir=str(cache), verbose=False)
    monkeypatch.setattr(wind_local._cache, 'require',
                        lambda name, *rest: cache.joinpath(*rest))
    wind_local._CLIM.clear()
    assert wind_local.climatology_period() == '1991-2020 (climatology)'


def test_the_two_surfaces_keep_their_own_prefixes():
    """`fetch_environment` describes two different objects that both live at
    the top of the water column, and their names used to cross over.

    The **ice boundary** is `surface`, `surface_sources`,
    `range_dependent_surface`, `surface_n_points`. The **wave realisation**
    is `altimetry`, `altimetry_sources` — and used to be
    `sea_surface_n_points`, `sea_surface_seed`, so the wave group changed its
    own prefix halfway and landed `sea_surface_n_points` next to a different
    object's `surface_n_points`. Reaching for the obvious name was accepted,
    did nothing to the waves, and warned about nothing.
    """
    import inspect
    from uacpy.data.environment import fetch_environment

    names = set(inspect.signature(fetch_environment).parameters)
    assert {'surface', 'surface_sources', 'range_dependent_surface',
            'surface_n_points'} <= names
    assert {'altimetry', 'altimetry_sources', 'altimetry_n_points',
            'altimetry_rng'} <= names
    crossover = {n for n in names if n.startswith('sea_surface')}
    assert not crossover, (
        f'{sorted(crossover)} uses a third prefix for the wave realisation, '
        f'which is spelled altimetry_* everywhere else in this signature')
