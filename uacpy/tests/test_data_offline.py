"""Tests for the offline local-cache backend (uacpy.data._cache + readers).

Builds a synthetic ``$UACPY_DATA_CACHE`` (tiny GEBCO/WOA23 NetCDF + sediment
CSVs) so the GEBCO/WOA23/DECK41 readers (``source='local'``) and
``fetch_environment(prefer_cache=True)`` run from the cache. Skipped where
netCDF4 is unavailable (the grids need it).
"""

import numpy as np
import pytest

netCDF4 = pytest.importorskip('netCDF4')

import uacpy.data as data
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import (
    _cache, emodnet_local, gebco_local, sediment_db, sound_speed, woa23_local,
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
    mk('t00', [18, 16, 13, 8, 5.0])
    mk('s00', [36, 36.1, 36.2, 35.5, 35.0])


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


def test_bathymetry_source_local(cache):
    assert data.fetch_point_depth((12.0, 34.0), source='local') == 1500.0
    t = data.fetch_transect((1.0, 1.0), (2.0, 2.0), n_points=4, source='local')
    assert t.shape == (4, 2) and np.all(t[:, 1] == 1500.0)


def test_bathymetry_bad_source_raises():
    with pytest.raises(ConfigurationError, match='source'):
        data.fetch_point_depth((1.0, 2.0), source='nope')


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


# ── capstone offline ────────────────────────────────────────────────────────

def test_fetch_environment_prefer_cache(cache):
    # prefer_cache=True resolves entirely from the installed cache (no network).
    env = data.fetch_environment((30.5, -40.5), prefer_cache=True, bottom='auto')
    assert env.depth == 1500.0
    assert env.ssp.n_depths >= 5
    assert env.bottom.grain_size_phi == 3.0
    assert [s.id for s in env.data_sources] == ['gebco', 'woa23', 'grainsize']


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
