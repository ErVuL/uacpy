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
    # Inverts the Pierson-Moskowitz Hs = coeff·U², so it must round-trip
    # against whatever coefficient the module carries.
    for u in (5.0, 10.0, 18.0):
        hs = sea_surface._PM_HS_COEFF * u ** 2
        assert sea_surface.hs_to_pm_wind(hs) == pytest.approx(u)
    assert sea_surface.hs_to_pm_wind(-1.0) == 0.0        # clamped, not NaN


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


def test_altimetry_guard_falls_back_to_literal():
    # altimetry= is the documented fallback when the sea-state fetch cannot
    # run; a missing date= must reach that fallback, not raise past it.
    alt = np.column_stack([np.linspace(0.0, 5000.0, 10), np.zeros(10)])
    env = env_mod.fetch_environment(
        (50.0, 0.0), bathymetry=2000.0, ssp=1500.0, transect_to=(50.5, 0.5),
        altimetry_sources='waves', altimetry=alt)
    assert env.altimetry is not None


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


def _stub_waves(monkeypatch, hs):
    monkeypatch.setattr(
        waves_mod, 'fetch_waves',
        lambda point, **kw: {'hs': hs, 'tp': 8.0, 'source': 'waverys'})


@pytest.mark.parametrize('max_range', [1e3, 1e4, 5e4, 1e5, 4.27e5])
def test_sea_surface_holds_its_wave_height_at_every_transect_length(
        max_range, monkeypatch):
    """The realization is sized from the sea state, so its significant wave
    height tracks the fetched one however long the transect is.

    With a fixed 500 samples the range step outgrows the Pierson-Moskowitz
    peak wavelength (64 m at U = 10 m/s) and the whole spectrum falls above
    Nyquist: the realized Hs was 86 % of the requested one over 10 km, 2.7 %
    over 50 km and numerically zero over 427 km — a silently flat sea.
    """
    hs = 2.1
    _stub_waves(monkeypatch, hs)
    alt, src = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', max_range=max_range, seed=7)
    assert src == 'waverys'
    assert alt[0, 0] == 0.0 and alt[-1, 0] == pytest.approx(max_range)
    # Hs = 4·rms for a Gaussian sea surface.
    assert 4.0 * np.std(alt[:, 1]) == pytest.approx(hs, rel=0.1)
    # The step resolves the peak wavelength lambda_p = 2*pi*U^2/g.
    u = sea_surface.hs_to_pm_wind(hs)
    dx = max_range / (len(alt) - 1)
    assert dx <= 2 * np.pi * u ** 2 / sea_surface._G / sea_surface._SAMPLES_PER_PEAK


def test_sea_surface_keeps_the_historical_floor_on_a_short_transect(monkeypatch):
    # 1 km at U = 10 m/s needs only ~130 samples to resolve the peak; the
    # realization keeps the old fixed default as a floor so a short transect
    # does not come back coarser than it used to.
    _stub_waves(monkeypatch, 2.1)
    alt, _ = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', max_range=1000.0, seed=1)
    assert len(alt) == sea_surface._MIN_SURFACE_POINTS


def test_pinned_n_points_too_coarse_to_resolve_the_peak_warns(monkeypatch):
    """A caller-pinned count that aliases the spectrum away warns, naming the
    step it produces and the count that would resolve the peak, instead of
    returning a flat surface silently."""
    _stub_waves(monkeypatch, 2.1)
    with pytest.warns(UserWarning, match=r'dx = 100\.2 m.*n_points >= 6\d{3}'):
        alt, _ = sea_surface.fetch_sea_surface(
            (50.0, 0.0), date='2020-01-01', max_range=50000.0, n_points=500,
            seed=1)
    assert alt.shape == (500, 2)              # the pinned count is still honoured


def test_sea_surface_sizing_is_capped_and_warned_for_a_calm_long_transect(
        monkeypatch):
    # A near-calm sea has a short peak wavelength, so resolving it over a long
    # transect would ask for millions of samples: the count is capped and the
    # shortfall reported rather than allocated.
    _stub_waves(monkeypatch, 0.09)                       # U ~ 2 m/s
    with pytest.warns(UserWarning, match='cap'):
        alt, _ = sea_surface.fetch_sea_surface(
            (50.0, 0.0), date='2020-01-01', max_range=4.27e5, seed=1)
    assert len(alt) == sea_surface._MAX_SURFACE_POINTS


def test_empty_wave_source_raises_a_typed_error():
    with pytest.raises(ConfigurationError, match='No data source was tried'):
        waves_mod.fetch_waves((50.0, 0.0), date='2020-01-01', source=())


def test_nbs_monthly_grid_maps_a_filled_cell_to_nan(monkeypatch, tmp_path):
    """Same hazard on the wind climatology build: a masked cell must not enter
    the monthly mean as a real wind speed."""
    netCDF4 = pytest.importorskip('netCDF4')
    path = tmp_path / 'wind.nc'
    ds = netCDF4.Dataset(path, 'w')
    for name, size in (('time', 1), ('altitude', 1),
                       ('latitude', 2), ('longitude', 2)):
        ds.createDimension(name, size)
    ds.createVariable('latitude', 'f8', ('latitude',))[:] = [0.5, 1.5]
    ds.createVariable('longitude', 'f8', ('longitude',))[:] = [0.5, 1.5]
    w = ds.createVariable('wind_speed', 'f4',
                          ('time', 'altitude', 'latitude', 'longitude'),
                          fill_value=-9999.0)
    field = np.ma.masked_array(np.full((1, 1, 2, 2), 7.0))
    field[0, 0, 0, 1] = np.ma.masked
    w[:] = field
    ds.close()
    blob = path.read_bytes()
    monkeypatch.setattr(wind_local, 'http_get', lambda url, **kw: blob)

    _, _, speed = wind_local._fetch_monthly_grid(2020, 3, timeout=1.0,
                                                 verbose=False)
    assert np.isnan(speed[0, 1])
    assert speed[0, 0] == pytest.approx(7.0)


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


def test_wind_climatology_closes_its_handle_when_a_variable_is_missing(
        monkeypatch):
    from uacpy.data import _netcdf, wind_local
    opened = []

    def fake_open(path):
        ds = _RecordingDataset()
        opened.append(ds)
        return ds

    monkeypatch.setattr(_netcdf, 'open_netcdf', fake_open)
    monkeypatch.setattr(wind_local, 'http_get', lambda url, **kw: b'not-a-grid')
    assert wind_local._fetch_monthly_grid(2000, 1, timeout=1.0,
                                          verbose=False) is None
    assert opened and all(ds.closed for ds in opened), (
        "the handle leaks once per month a 120-grid climatology build cannot "
        "name a variable in")


def test_two_empty_month_sweeps_stop_the_wind_climatology_build(
        monkeypatch, tmp_path):
    """Zero grids across the first two month sweeps raise the typed error
    before the remaining ten months retry their way to the same place."""
    import uacpy.data.wind_local as wl
    calls = []
    def nothing(year, month, *, timeout, verbose):
        calls.append((year, month))
        return None
    monkeypatch.setattr(wl, '_fetch_monthly_grid', nothing)
    with pytest.raises(DataFetchError,
                       match='stopping before the remaining ten months'):
        wl.download_wind_db(cache_dir=tmp_path, years=range(2013, 2023))
    assert len(calls) == 20


# ── climatology vintage (the reference period the cache records) ─────────────

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
    """The cache used to store lat/lon/speed only, so a build's reference
    period was unrecoverable from the file."""
    assert wind_local.climatology_period() == '2013-2022 (climatology)'


def test_a_wind_cache_without_a_period_loads_its_grid(wind_cache):
    """Old caches and the synthetic ones the tests build carry no ``years``.
    Absent is not an error: the grid still reads, the vintage is unstated."""
    assert wind_local.climatology_period() is None
    assert wind_local.wind_speed((0.6, 0.6), date='2021-03-15') == pytest.approx(8.5)


def test_a_built_wind_cache_records_its_years(tmp_path, monkeypatch):
    """``download_wind_db`` writes the period it averaged. The per-month grid
    fetch is stubbed, so this never touches the network."""
    root = tmp_path / 'built_wind'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    wind_local._CLIM.clear()
    lat = np.linspace(-89.875, 89.875, 8)
    lon = np.linspace(0.0, 359.75, 8)
    calls = []

    def _stub_grid(year, month, **_kwargs):
        calls.append((year, month))
        return lat, lon, np.full((lat.size, lon.size), 6.25)

    monkeypatch.setattr(wind_local, '_fetch_monthly_grid', _stub_grid)
    out = wind_local.download_wind_db(years=(2019, 2020, 2021))
    assert calls, 'the stubbed month fetch was never reached'
    with np.load(out, allow_pickle=False) as data:
        assert [int(y) for y in data['years']] == [2019, 2020, 2021]
    assert wind_local.climatology_period() == '2019-2021 (climatology)'


def test_the_environment_provenance_carries_the_wind_vintage(dated_wind_cache):
    """``_climatology_vintage`` is what puts the period on the provenance
    record; every other source and every period-less cache yield None."""
    from uacpy.data.environment import _climatology_vintage
    assert _climatology_vintage('wind') == '2013-2022 (climatology)'
    assert _climatology_vintage('gebco') is None
    assert _climatology_vintage('woa23') is None
