"""Tests for the offline local-cache backend (uacpy.data._cache + readers).

Builds a synthetic ``$UACPY_DATA_CACHE`` (tiny GEBCO/WOA23 NetCDF + sediment
CSVs) so the GEBCO/WOA23/DECK41 readers (``source='local'``) and a cache-first
``fetch_environment`` (gebco/woa23 + ``bottom_sources='grainsize'``) run from the
cache with no network. Skipped where netCDF4 is unavailable (the grids need it).
"""

import numpy as np
import pytest

netCDF4 = pytest.importorskip('netCDF4')

import uacpy.data as data
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import (
    _cache, crust1_local, emodnet_local, gebco_local, globsed_local, sediment_db,
    sound_speed, woa23_local,
)

_FILL = 9.96921e36


def _write_gebco(cache, *, deep=-1500.0, land_at=None):
    gdir = cache / 'gebco'; gdir.mkdir(parents=True)
    lat = np.arange(-90, 91, 1.0)
    lon = np.arange(-180, 180, 1.0)
    elev = np.full((lat.size, lon.size), deep)
    if land_at is not None:
        i = int(round(land_at[0] + 90)); j = int(round(land_at[1] + 180))
        elev[i, j] = 25.0
    ds = netCDF4.Dataset(gdir / 'GEBCO_2025.nc', 'w')
    ds.createDimension('lat', lat.size); ds.createDimension('lon', lon.size)
    ds.createVariable('lat', 'f8', ('lat',))[:] = lat
    ds.createVariable('lon', 'f8', ('lon',))[:] = lon
    ds.createVariable('elevation', 'f4', ('lat', 'lon'))[:] = elev
    ds.close()


def _write_woa(cache):
    wdir = cache / 'woa23'; wdir.mkdir(parents=True)
    depth = np.array([0, 50, 100, 500, 1000.0])
    # column for grid_index(30.5, -40.5) == (120, 139) on the 1° grid.
    def mk(var, vals):
        arr = np.full((1, depth.size, 180, 360), _FILL)
        arr[0, :, 120, 139] = vals
        ds = netCDF4.Dataset(wdir / f'woa23_decav_{var}_01.nc', 'w')
        for d, n in [('time', 1), ('depth', depth.size), ('lat', 180), ('lon', 360)]:
            ds.createDimension(d, n)
        ds.createVariable('depth', 'f4', ('depth',))[:] = depth
        name = 't_an' if var[0] == 't' else 's_an'
        ds.createVariable(name, 'f4', ('time', 'depth', 'lat', 'lon'))[:] = arr
        ds.close()
    # Period 00 = annual (date-less tests); 03 = March, so the sea-ice tests
    # (which require a date) resolve WOA from the cache instead of falling
    # through to a live OPeNDAP request.
    for period in ('00', '03'):
        mk(f't{period}', [18, 16, 13, 8, 5.0])
        mk(f's{period}', [36, 36.1, 36.2, 35.5, 35.0])


def _write_sediment(cache):
    sdir = cache / 'sediment'; sdir.mkdir(parents=True)
    (sdir / 'grainsize.csv').write_text(
        'latitude,longitude,mean_phi\n30.5,-40.5,3.0\n43.0,7.0,2.0\n')
    (sdir / 'deck41.csv').write_text(
        'latitude,longitude,lithology\n44.0,8.0,Sand\n10.0,20.0,Clay\n')


def _write_emodnet(cache):
    """One synthetic Folk-5 Sand polygon over the North Sea; skipped sans shapely.

    Deliberately *not* over the grain-size test point (30.5, -40.5) so 'auto'
    there exercises the EMODnet-miss → grain-size fallthrough.
    """
    try:
        import pickle
        import shapely
    except ImportError:                              # pragma: no cover
        return
    edir = cache / 'emodnet'; edir.mkdir(parents=True)
    poly = shapely.geometry.box(2.0, 54.0, 3.0, 56.0)       # lon0,lat0,lon1,lat1
    with open(edir / 'seabed_substrate.pkl', 'wb') as fh:
        pickle.dump({'codes': [2], 'wkb': [shapely.to_wkb(poly)]}, fh)


def _write_globsed(cache):
    gdir = cache / 'globsed'; gdir.mkdir(parents=True)
    ds = netCDF4.Dataset(gdir / 'GlobSed-v3.nc', 'w')
    ds.createDimension('lat', 181); ds.createDimension('lon', 361)
    ds.createVariable('lat', 'f8', ('lat',))[:] = np.linspace(-90, 90, 181)
    ds.createVariable('lon', 'f8', ('lon',))[:] = np.linspace(-180, 180, 361)
    ds.createVariable('z', 'f4', ('lat', 'lon'))[:] = 500.0      # uniform 500 m
    ds.close()


def _write_crust1(cache):
    cdir = cache / 'crust1'; cdir.mkdir(parents=True)
    n = 180 * 360
    rows = {                                       # uniform ocean column, 1 km sed
        'crust1.bnds': [0, -4, -4, -5, -5, -5, -10, -20, -30],
        'crust1.vp':   [1.5, 3.8, 2.0, 0, 0, 5.0, 6.5, 7.1, 8.1],
        'crust1.vs':   [0, 1.9, 0.6, 0, 0, 2.7, 3.7, 4.0, 4.5],
        'crust1.rho':  [1.02, 0.9, 1.9, 0, 0, 2.6, 2.8, 3.0, 3.3],
    }
    # Every cell is identical, so repeat one formatted line — far cheaper than
    # np.savetxt over the full 64800×9 grid in this per-test fixture.
    for name, row in rows.items():
        line = ' '.join('%g' % v for v in row) + '\n'
        (cdir / name).write_text(line * n)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A synthetic, fully-populated offline cache wired via $UACPY_DATA_CACHE."""
    root = tmp_path / 'data_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    # Module-level open-once caches are keyed by path; clear for isolation.
    gebco_local._GRID.clear()
    woa23_local._DATASETS.clear()
    sediment_db._SAMPLES.clear()
    emodnet_local._INDEX.clear()
    _write_gebco(root, land_at=(0.0, 0.0))
    _write_woa(root)
    _write_sediment(root)
    _write_emodnet(root)
    return root


@pytest.fixture
def seismic_cache(tmp_path, monkeypatch):
    """Cache with the GlobSed + CRUST1.0 layered-bottom stack (and GEBCO/WOA).

    Kept separate from ``cache`` so the bulky GlobSed grid / CRUST1.0 columns are
    built only for the one capstone test that needs the layered seabed, not on
    every offline test.
    """
    root = tmp_path / 'seismic_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    gebco_local._GRID.clear()
    woa23_local._DATASETS.clear()
    crust1_local._MODEL.clear()
    globsed_local._GRID.clear()
    _write_gebco(root)
    _write_woa(root)
    _write_globsed(root)
    _write_crust1(root)
    return root


# ── cache resolver ──────────────────────────────────────────────────────────

def test_cache_root_honors_env(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'x'))
    assert _cache.cache_root() == tmp_path / 'x'


def test_require_missing_names_install_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    with pytest.raises(ConfigurationError, match='install.sh --data gebco'):
        _cache.require('gebco')


# ── GEBCO local ─────────────────────────────────────────────────────────────

def test_gebco_point_depth_and_land(cache):
    assert gebco_local.point_depth((10.0, 20.0)) == 1500.0
    with pytest.raises(DataFetchError, match='land'):
        gebco_local.point_depth((0.0, 0.0))


def test_gebco_region_grid_marks_land_nan(cache):
    lats, lons, depth = gebco_local.region_grid((-3, 3), (-3, 3), 7, 7)
    assert depth.shape == (7, 7)
    assert np.isnan(depth).sum() == 1            # the single land cell
    assert np.nanmin(depth) == 1500.0


def test_bathymetry_sources_local(cache):
    assert data.fetch_bathy((12.0, 34.0), source='local') == 1500.0
    t = data.fetch_bathy_transect((1.0, 1.0), (2.0, 2.0), n_points=4, source='local')
    assert t.shape == (4, 2) and np.all(t[:, 1] == 1500.0)


def test_bathymetry_bad_source_raises():
    with pytest.raises(ConfigurationError, match='source'):
        data.fetch_bathy((1.0, 2.0), source='nope')


# ── WOA23 local ─────────────────────────────────────────────────────────────

def test_woa_local_ssp(cache):
    prof = sound_speed.fetch_ssp((30.5, -40.5), source='local')
    assert prof.depths.tolist() == [0.0, 50.0, 100.0, 500.0, 1000.0]
    assert np.all((1450 < prof.data[:, 0]) & (prof.data[:, 0] < 1560))


def test_woa_local_ts_column(cache):
    # The local column feeds the *same* UNESCO conversion as OPeNDAP; check the
    # raw T/S surface values come through untouched.
    z, t, s = sound_speed.fetch_ts_profile((30.5, -40.5), source='local')
    assert t[0] == 18.0 and s[0] == 36.0 and z[0] == 0.0


# ── sediment local ──────────────────────────────────────────────────────────

def test_sediment_sample_and_bottom(cache):
    s = sediment_db.fetch_sediment_sample((30.51, -40.51))
    assert s['phi'] == 3.0 and s['distance_km'] < 5.0
    bp = sediment_db.fetch_bottom_local((30.51, -40.51))
    assert bp.acoustic_type == 'half-space' and bp.grain_size_phi == 3.0


def test_sediment_deck41_lithology_path(cache):
    bp = sediment_db.fetch_bottom_local((44.01, 8.01))
    assert bp.grain_size_phi == 1.5            # 'Sand' → ϕ 1.5


def test_download_sediment_db_normalizes(tmp_path, monkeypatch):
    # Build a minimal G00127-style tarball (sample.txt + phi.txt) and stub the
    # download; the normalizer must join lat/lon with a weighted-mean ϕ.
    import io
    import tarfile

    sample = ("mggid\tsample\tlat\tlon\n"
              "A\t1\t40.0\t-60.0\n"
              "B\t1\t10.0\t20.0\n")
    # A: 50% in ϕ[1,2] + 50% in ϕ[3,4] → mean 2.5 ; B: all in ϕ[7,8] → 7.5
    phi = ("mggid\tsample\tinterval\tlower_phi_limit\tupper_phi_limit\tweight_percent\n"
           "A\t1\t0\t1\t2\t50\n"
           "A\t1\t0\t3\t4\t50\n"
           "B\t1\t0\t7\t8\t100\n")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for name, text in [('g00127sample.txt', sample), ('g00127phi.txt', phi)]:
            data = text.encode()
            info = tarfile.TarInfo(name); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(sediment_db, 'http_get', lambda url, **kw: buf.getvalue())

    out = sediment_db.download_sediment_db(cache_dir=str(tmp_path))
    import csv
    rows = {r['latitude']: r for r in csv.DictReader(open(out))}
    assert float(rows['40.0']['mean_phi']) == pytest.approx(2.5)
    assert float(rows['10.0']['mean_phi']) == pytest.approx(7.5)


def test_sediment_too_far_raises(cache):
    with pytest.raises(DataFetchError, match='km away'):
        sediment_db.fetch_sediment_sample((-80.0, 150.0), max_distance_km=100)


def test_sediment_transect(cache):
    rdb = sediment_db.fetch_bottom_local_transect(
        (30.5, -40.5), (43.0, 7.0), n_points=4)
    assert rdb.sound_speed.shape == (4,)


# ── capstone from the cache ───────────────────────────────────────────────────
# These pin cache-only sources (gebco/woa23 cache-first, bottom_sources='grainsize'
# /'crust1'), so every fetch resolves locally with no network call.

def test_fetch_environment_from_cache(cache):
    # Cache-first: gebco/woa23 resolve to their local twin, grainsize to the
    # installed sediment DB — no network, fully reproducible.
    env = data.fetch_environment((30.5, -40.5), bottom_sources='grainsize')
    assert env.depth == 1500.0
    assert env.ssp.n_depths >= 5
    assert env.bottom.grain_size_phi == 3.0
    assert [s.id for s in env.data_sources] == ['gebco', 'woa23', 'grainsize']


def test_fetch_environment_cache_preset(cache):
    # *_sources='cache' resolves every axis from local data only (no network):
    # bathy→GEBCO-local, ssp→WOA23-local, bottom→cache chain (EMODnet local miss
    # here → grain-size DB).
    env = data.fetch_environment((30.5, -40.5), bathymetry_sources='cache',
                                 ssp_sources='cache', bottom_sources='cache')
    assert env.depth == 1500.0
    assert env.bottom.grain_size_phi == 3.0
    assert [s.id for s in env.data_sources] == ['gebco', 'woa23', 'grainsize']


def test_cache_preset_never_hits_network(tmp_path, monkeypatch):
    # 'cache' must never reach the network. With an empty cache and literal
    # bathy/ssp, the bottom 'cache' chain (incl. the pelagic last resort, whose
    # depth lookup would otherwise fall back to the live API) must fail fast with
    # the install hint rather than touching the net.
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    gebco_local._GRID.clear()
    from uacpy.data import bathymetry
    monkeypatch.setattr(bathymetry, '_fetch_depths', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("ssp_sources/bottom_sources='cache' hit the live API")))
    with pytest.raises(ConfigurationError, match='install.sh --data'):
        data.fetch_environment((30.5, -40.5), bathymetry=3000.0, ssp=1500.0,
                               bottom_sources='cache')


def test_cache_preset_ssp_no_cache_raises(tmp_path, monkeypatch):
    # ssp_sources='cache' with no WOA23 cache fails fast (no OpenDAP fallback).
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    woa23_local._DATASETS.clear()
    with pytest.raises(ConfigurationError, match='install.sh --data woa23'):
        data.fetch_environment((30.5, -40.5), bathymetry=3000.0,
                               ssp_sources='cache')


def test_fetch_environment_crust1_pulls_globsed(seismic_cache):
    # bottom_sources='crust1' rescales its column with GlobSed by default, so
    # both CRUST1.0 and GlobSed appear in the environment's provenance.
    env = data.fetch_environment((30.5, -40.5), bottom_sources='crust1')
    ids = [s.id for s in env.data_sources]
    assert ids == ['gebco', 'woa23', 'crust1', 'globsed']
    assert env.bottom.total_thickness() == pytest.approx(500.0)


def test_fetch_environment_sea_ice(cache, monkeypatch):
    # surface_sources='seaice' sets the surface from the climatological
    # concentration; an ice-covered point gets an elastic canopy + provenance.
    from uacpy.data import seaice_local
    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration',
                        lambda pt, date=None, *, month=None: 0.85)
    env = data.fetch_environment((30.5, -40.5), date='2026-03-15',
                                 bottom_sources='grainsize',
                                 surface_sources='seaice')
    assert env.surface.acoustic_type == 'half-space'
    assert env.surface.shear_speed == 1800.0 and env.has_elastic_surface()
    assert 'seaice' in [s.id for s in env.data_sources]


def test_fetch_environment_sea_ice_open_water(cache, monkeypatch):
    # Below the ice-edge → free surface untouched, no 'seaice' provenance.
    from uacpy.data import seaice_local
    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration',
                        lambda pt, date=None, *, month=None: 0.0)
    env = data.fetch_environment((30.5, -40.5), date='2026-03-15',
                                 bottom_sources='grainsize',
                                 surface_sources='auto')
    assert env.surface.acoustic_type == 'vacuum'
    assert 'seaice' not in [s.id for s in env.data_sources]


def test_fetch_environment_sea_ice_requires_date(cache):
    with pytest.raises(ConfigurationError, match="surface_sources='seaice' needs date"):
        data.fetch_environment((30.5, -40.5), surface_sources='seaice')


def test_offline_emodnet_local_bottom(cache):
    pytest.importorskip('shapely')
    bp = emodnet_local.fetch_bottom_local((55.0, 2.5))     # inside the polygon
    assert bp.acoustic_type == 'half-space' and bp.grain_size_phi == 2.0
    with pytest.raises(DataFetchError, match='European seas only'):
        emodnet_local.fetch_seabed_local((30.5, -40.5))    # outside it


def test_offline_emodnet_missing_names_flag(tmp_path, monkeypatch):
    pytest.importorskip('shapely')
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    emodnet_local._INDEX.clear()
    with pytest.raises(ConfigurationError, match='install.sh --data emodnet'):
        emodnet_local.fetch_bottom_local((56.0, 3.0))
