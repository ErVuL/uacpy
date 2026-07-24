"""Tests for the low-frequency seabed backends: GlobSed + CRUST1.0.

Builds a tiny synthetic ``$UACPY_DATA_CACHE`` (1° GlobSed grid + a uniform-ocean
CRUST1.0 column) so the thickness reader and the layered-elastic bottom builder
run fully offline. Skipped where netCDF4 (the grid dependency) is unavailable.
"""

import io
import tarfile
import warnings

import numpy as np
import pytest

netCDF4 = pytest.importorskip('netCDF4')

from uacpy.core.environment import SeabedColumn, Bottom
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import crust1_local, globsed_local


def _write_globsed(root):
    gdir = root / 'globsed'; gdir.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(-180, 180, 361)
    z = np.full((181, 361), 500.0)
    ds = netCDF4.Dataset(gdir / 'GlobSed-v3.nc', 'w')
    ds.createDimension('lat', 181); ds.createDimension('lon', 361)
    ds.createVariable('lat', 'f8', ('lat',))[:] = lat
    ds.createVariable('lon', 'f8', ('lon',))[:] = lon
    zv = ds.createVariable('z', 'f4', ('lat', 'lon'), fill_value=np.nan)
    z[0, 0] = np.nan                                   # one no-data cell
    zv[:] = z
    ds.close()


def _write_crust1(root):
    cdir = root / 'crust1'; cdir.mkdir(parents=True, exist_ok=True)
    n = 180 * 360
    # Uniform ocean column: water 0→−4 km, 1 km sediment (−4→−5), crust below.
    rows = {
        'crust1.bnds': [0, -4, -4, -5, -5, -5, -10, -20, -30],
        'crust1.vp':   [1.5, 3.8, 2.0, 0, 0, 5.0, 6.5, 7.1, 8.1],
        'crust1.vs':   [0, 1.9, 0.6, 0, 0, 2.7, 3.7, 4.0, 4.5],
        'crust1.rho':  [1.02, 0.9, 1.9, 0, 0, 2.6, 2.8, 3.0, 3.3],
    }
    for name, row in rows.items():
        np.savetxt(cdir / name, np.tile(row, (n, 1)), fmt='%g')


@pytest.fixture(scope='module')
def _seis_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('seis_cache')
    _write_globsed(root)
    _write_crust1(root)
    return root


@pytest.fixture
def cache(_seis_root, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(_seis_root))
    globsed_local._GRID.clear()
    crust1_local._MODEL.clear()
    return _seis_root


# ── GlobSed ──────────────────────────────────────────────────────────────────

def test_globsed_thickness(cache):
    assert globsed_local.fetch_sediment_thickness((30.0, -40.0)) == 500.0


def test_globsed_no_data_raises(cache):
    with pytest.raises(DataFetchError, match='no sediment thickness'):
        globsed_local.fetch_sediment_thickness((-90.0, -180.0))   # the NaN cell


def test_globsed_transect(cache):
    r, thk = globsed_local.fetch_sediment_thickness_transect(
        (30.0, -40.0), (31.0, -40.0), n_points=3)
    assert r.shape == (3,) and thk.shape == (3,) and np.allclose(thk, 500.0)


# ── CRUST1.0 ─────────────────────────────────────────────────────────────────

def test_crust1_profile(cache):
    p = crust1_local.fetch_crust1_profile((30.0, -40.0))
    assert p['water_depth_m'] == 4000.0
    assert p['sediment_thickness_m'] == 1000.0


def test_crust1_bottom_layered_elastic(cache):
    b = crust1_local.fetch_bottom_crust1((30.0, -40.0))
    assert isinstance(b, SeabedColumn)
    assert b.layers[0].sound_speed == 2000.0       # 2.0 km/s → m/s
    assert b.layers[0].shear_speed == 600.0        # 0.6 km/s → m/s (elastic)
    assert b.halfspace.sound_speed == 5000.0       # upper crystalline crust


def test_crust1_fluid_option(cache):
    b = crust1_local.fetch_bottom_crust1((30.0, -40.0), elastic=False)
    assert all(layer.shear_speed == 0 for layer in b.layers)
    assert b.halfspace.shear_speed == 0


def test_crust1_thickness_rescale(cache):
    b = crust1_local.fetch_bottom_crust1((30.0, -40.0), sediment_thickness=2000.0)
    assert b.total_thickness() == pytest.approx(2000.0)
    assert b.sediment_thickness_source is None      # explicit, not GlobSed


def test_crust1_uses_globsed_by_default(cache):
    # GlobSed (500 m here) rescales CRUST1.0's native 1000 m column by default,
    # and the bottom records that GlobSed supplied the thickness.
    b = crust1_local.fetch_bottom_crust1((30.0, -40.0))
    assert b.total_thickness() == pytest.approx(500.0)
    assert b.sediment_thickness_source == 'globsed'


def test_crust1_globsed_fallback_keeps_native(cache):
    # Where GlobSed has no value, CRUST1.0's own thickness is kept (no marker).
    b = crust1_local.fetch_bottom_crust1((-90.0, -180.0))   # GlobSed NaN cell
    assert b.total_thickness() == pytest.approx(1000.0)
    assert b.sediment_thickness_source is None


def test_crust1_use_globsed_false(cache):
    b = crust1_local.fetch_bottom_crust1((30.0, -40.0), use_globsed=False)
    assert b.total_thickness() == pytest.approx(1000.0)
    assert b.sediment_thickness_source is None


def test_crust1_transect(cache):
    rdl = crust1_local.fetch_bottom_crust1_transect(
        (30.0, -40.0), (31.0, -40.0), n_points=3)
    assert isinstance(rdl, Bottom)
    assert len(rdl.columns) == 3 and rdl.ranges[0] == 0.0
    assert rdl.sediment_thickness_source == 'globsed'   # GlobSed applied per point


def test_crust1_transect_accepts_max_points(cache):
    # environment._fetch_bottom forwards max_points to every transect bottom
    # fetcher; crust1's must accept it (it used to raise TypeError) and clamp
    # n_points to it.
    rdl = crust1_local.fetch_bottom_crust1_transect(
        (30.0, -40.0), (31.0, -40.0), n_points=6, max_points=3)
    assert isinstance(rdl, Bottom)
    assert len(rdl.columns) == 3                         # clamped to max_points


def test_crust1_emits_commercial_warning(cache):
    # CRUST1.0 is non-commercial (no formal licence); the low-level fetcher must
    # warn so a direct caller is never silent, and the transect fetcher must
    # warn exactly once (not once per waypoint).
    with pytest.warns(UserWarning, match='commercial'):
        crust1_local.fetch_bottom_crust1((30.0, -40.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        crust1_local.fetch_bottom_crust1_transect(
            (30.0, -40.0), (31.0, -40.0), n_points=4)
    commercial = [w for w in caught
                  if issubclass(w.category, UserWarning)
                  and 'commercial' in str(w.message)]
    assert len(commercial) == 1


def test_crust1_missing_cache_names_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    crust1_local._MODEL.clear()
    with pytest.raises(ConfigurationError, match='install.sh --data crust1'):
        crust1_local.fetch_bottom_crust1((30.0, -40.0))


# ── downloads (mocked) ───────────────────────────────────────────────────────

def test_globsed_download(tmp_path, monkeypatch):
    src = tmp_path / 'src.nc'
    ds = netCDF4.Dataset(src, 'w')
    ds.createDimension('lat', 2); ds.createDimension('lon', 2)
    ds.createVariable('lat', 'f8', ('lat',))[:] = [0, 1]
    ds.createVariable('lon', 'f8', ('lon',))[:] = [0, 1]
    ds.createVariable('z', 'f4', ('lat', 'lon'))[:] = 1.0
    ds.close()
    blob = src.read_bytes()
    # Force the urllib fallback (no real curl to NCEI) and mock the fetch.
    monkeypatch.setattr(globsed_local, '_curl_download', lambda *a, **k: False)
    monkeypatch.setattr(globsed_local, 'http_get', lambda url, **kw: blob)
    out = globsed_local.download_globsed_db(cache_dir=str(tmp_path / 'o'))
    assert out.exists() and out.name == 'GlobSed-v3.nc'


def test_crust1_download(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for name in ('crust1.bnds', 'crust1.vp', 'crust1.vs', 'crust1.rho'):
            blob = b'0 0 0 0 0 0 0 0 0\n'
            info = tarfile.TarInfo(name); info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))
    monkeypatch.setattr(crust1_local, 'http_get', lambda url, **kw: buf.getvalue())
    out = crust1_local.download_crust1_db(cache_dir=str(tmp_path / 'o'))
    assert (out / 'crust1.bnds').exists()


def test_globsed_interrupted_curl_no_final_file(tmp_path, monkeypatch):
    """A curl killed mid-transfer must not poison the cache with a truncated
    grid at the final path."""
    from pathlib import Path
    dest = tmp_path / 'o'

    def fake_run(cmd, **kw):
        Path(cmd[cmd.index('-o') + 1]).write_bytes(b'truncated')
        raise KeyboardInterrupt

    monkeypatch.setattr(globsed_local.shutil, 'which', lambda n: '/usr/bin/curl')
    monkeypatch.setattr(globsed_local.subprocess, 'run', fake_run)
    with pytest.raises(KeyboardInterrupt):
        globsed_local.download_globsed_db(cache_dir=str(dest))
    assert not (dest / globsed_local.GLOBSED_FILE).exists()
