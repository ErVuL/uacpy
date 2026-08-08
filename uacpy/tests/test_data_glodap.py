"""Tests for the offline GLODAP seawater-pH backend (uacpy.data.glodap_local).

Builds a synthetic ``$UACPY_DATA_CACHE`` GLODAP NetCDF (tiny ``pH(depth, lat,
lon)`` grid on the mapped product's shifted longitude axis) so the pH reader and
a cache-first ``fetch_environment(with_absorption=True)`` run from the cache with
no network. Skipped where netCDF4 is unavailable (the grid needs it).
"""

import numpy as np
import pytest

from uacpy.data import _cache

netCDF4 = pytest.importorskip('netCDF4')

import uacpy.data as data
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import gebco_local, glodap_local, sediment_db, woa23_local
from uacpy.data import _http

_FILL = 9.96921e36


def _write_glodap(cache):
    """Synthetic GLODAP pH grid on the mapped product's 20.5°→379.5° lon axis.

    One populated column at (30.5, -40.5) with a NaN-masked deepest level (a
    sub-seafloor fill), so the reader exercises both the shifted-origin longitude
    wrap and the seafloor trim; everywhere else is NaN (land / unmapped).
    """
    gdir = cache / 'glodap'; gdir.mkdir(parents=True)
    depth = np.array([0.0, 500.0, 2000.0])
    lat = np.arange(-89.5, 90.5, 1.0)
    lon = np.arange(20.5, 380.5, 1.0)               # 20.5 … 379.5 (shifted origin)
    ph = np.full((depth.size, lat.size, lon.size), np.nan)
    # col wrap: lon -40.5 → 319.5 (index 299); row: lat 30.5 → index 120.
    ph[:, 120, 299] = [8.10, 8.05, np.nan]
    ds = netCDF4.Dataset(gdir / glodap_local.GLODAP_FILE, 'w')
    ds.createDimension('depth', depth.size)
    ds.createDimension('lat', lat.size)
    ds.createDimension('lon', lon.size)
    ds.createVariable('Depth', 'f8', ('depth',))[:] = depth
    ds.createVariable('lat', 'f8', ('lat',))[:] = lat
    ds.createVariable('lon', 'f8', ('lon',))[:] = lon
    ds.createVariable('pHtsinsitutp', 'f4', ('depth', 'lat', 'lon'))[:] = ph
    ds.close()


def _write_gebco(cache, *, deep=-1500.0):
    gdir = cache / 'gebco'; gdir.mkdir(parents=True)
    lat = np.arange(-90, 91, 1.0)
    lon = np.arange(-180, 180, 1.0)
    ds = netCDF4.Dataset(gdir / 'GEBCO_2025.nc', 'w')
    ds.createDimension('lat', lat.size); ds.createDimension('lon', lon.size)
    ds.createVariable('lat', 'f8', ('lat',))[:] = lat
    ds.createVariable('lon', 'f8', ('lon',))[:] = lon
    ds.createVariable('elevation', 'f4', ('lat', 'lon'))[:] = deep
    ds.close()


def _write_woa(cache):
    wdir = cache / 'woa23'; wdir.mkdir(parents=True)
    depth = np.array([0, 50, 100, 500, 1000.0])
    def mk(var, vals):
        arr = np.full((1, depth.size, 180, 360), _FILL)
        arr[0, :, 120, 139] = vals                  # grid_index(30.5, -40.5)
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
        'latitude,longitude,mean_phi\n30.5,-40.5,3.0\n')


@pytest.fixture
def glodap_cache(tmp_path, monkeypatch):
    """Cache holding only the synthetic GLODAP pH grid."""
    root = tmp_path / 'glodap_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    _cache.invalidate_grids()
    _write_glodap(root)
    return root


@pytest.fixture
def full_cache(tmp_path, monkeypatch):
    """GEBCO + WOA23 + grain-size + GLODAP — a fully offline absorption run."""
    root = tmp_path / 'full_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    _cache.invalidate_grids()
    woa23_local._DATASETS.clear()
    sediment_db._SAMPLES.clear()
    _cache.invalidate_grids()
    _write_gebco(root)
    _write_woa(root)
    _write_sediment(root)
    _write_glodap(root)
    return root


# ── reader ──────────────────────────────────────────────────────────────────

def test_ph_profile_trims_at_seafloor(glodap_cache):
    depths, ph = glodap_local.fetch_ph_profile((30.5, -40.5))
    # The deepest (2000 m) level is a NaN fill → trimmed; two finite levels stay.
    assert depths.tolist() == [0.0, 500.0]
    assert ph.tolist() == [pytest.approx(8.10), pytest.approx(8.05)]


def test_ph_surface_and_reference_depth(glodap_cache):
    assert glodap_local.fetch_ph((30.5, -40.5)) == pytest.approx(8.10)
    assert glodap_local.fetch_ph((30.5, -40.5),
                                 reference_depth=500.0) == pytest.approx(8.05)


def test_ph_land_raises(glodap_cache):
    with pytest.raises(DataFetchError, match='land or unmapped'):
        glodap_local.fetch_ph_profile((0.0, 0.0))


def test_ph_missing_cache_names_install_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    _cache.invalidate_grids()
    with pytest.raises(ConfigurationError, match='install.sh --data glodap'):
        glodap_local.fetch_ph((30.5, -40.5))


# ── absorption integration ────────────────────────────────────────────────────

def test_with_absorption_uses_glodap_ph(full_cache):
    # A cached GLODAP grid replaces the default pH=8.1 with the real surface
    # value (8.10 here) and stamps 'glodap' provenance.
    env = data.fetch_environment((30.5, -40.5), ssp_sources='local',
                                 bottom_sources='grainsize',
                                 with_absorption=True)
    assert env.absorption.pH == pytest.approx(8.10)
    assert 'glodap' in [s.source.id for s in env.data_sources]


def test_with_absorption_no_glodap_keeps_default(tmp_path, monkeypatch):
    # Without the GLODAP cache, absorption silently keeps the model-default pH
    # and records no glodap provenance (cache-first, no hard failure).
    root = tmp_path / 'no_glodap'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    _cache.invalidate_grids()
    woa23_local._DATASETS.clear()
    sediment_db._SAMPLES.clear()
    _cache.invalidate_grids()
    _write_gebco(root)
    _write_woa(root)
    _write_sediment(root)
    from uacpy.data.absorption import DEFAULT_OCEAN_PH
    env = data.fetch_environment((30.5, -40.5), ssp_sources='local',
                                 bottom_sources='grainsize',
                                 with_absorption=True)
    assert env.absorption.pH == pytest.approx(DEFAULT_OCEAN_PH)
    assert 'glodap' not in [s.source.id for s in env.data_sources]


# ── download hygiene ─────────────────────────────────────────────────────────

def test_curl_interrupt_leaves_no_destination(tmp_path, monkeypatch):
    """A curl killed mid-transfer must never leave a truncated file at the
    final destination (the cache would accept it forever)."""
    from pathlib import Path
    out = tmp_path / 'GLODAPv2.tar.gz'

    def fake_run(cmd, **kw):
        Path(cmd[cmd.index('-o') + 1]).write_bytes(b'partial')
        raise KeyboardInterrupt

    monkeypatch.setattr(_http.shutil, 'which', lambda n: '/usr/bin/curl')
    monkeypatch.setattr(_http.subprocess, 'run', fake_run)
    with pytest.raises(KeyboardInterrupt):
        _http.curl_download('http://x', out, timeout=5.0, verbose=False)
    assert not out.exists()


def test_curl_failure_leaves_no_destination(tmp_path, monkeypatch):
    from pathlib import Path
    import subprocess
    out = tmp_path / 'GLODAPv2.tar.gz'

    def fake_run(cmd, **kw):
        Path(cmd[cmd.index('-o') + 1]).write_bytes(b'partial')
        raise subprocess.SubprocessError("curl died")

    monkeypatch.setattr(_http.shutil, 'which', lambda n: '/usr/bin/curl')
    monkeypatch.setattr(_http.subprocess, 'run', fake_run)
    assert _http.curl_download('http://x', out, timeout=5.0,
                                       verbose=False) is False
    assert not out.exists()


# ── tarball extraction guards ────────────────────────────────────────────────

def test_extract_ph_rejects_decompression_bomb(tmp_path):
    import gzip
    import tarfile
    info = tarfile.TarInfo(glodap_local.GLODAP_FILE)
    info.size = 3 * 1024 ** 3                    # over the 2 GiB member cap
    tar_path = tmp_path / 'bomb.tar.gz'
    with gzip.open(tar_path, 'wb') as f:
        f.write(info.tobuf(tarfile.GNU_FORMAT))
        f.write(b'\0' * 1024)
    with pytest.raises(DataFetchError, match='decompression bomb'):
        glodap_local._extract_ph(tar_path, tmp_path / 'out.nc')
    assert not (tmp_path / 'out.nc').exists()


def test_extract_ph_non_regular_member_raises(tmp_path):
    import tarfile
    tar_path = tmp_path / 'dir.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as tf:
        info = tarfile.TarInfo('mapped/' + glodap_local.GLODAP_FILE)
        info.type = tarfile.DIRTYPE
        tf.addfile(info)
    with pytest.raises(DataFetchError, match='not a regular file'):
        glodap_local._extract_ph(tar_path, tmp_path / 'out.nc')


def _write_glodap_fillvalue(cache, fill=-999.0):
    """GLODAP grid whose sub-seafloor level uses a real ``_FillValue``.

    The other fixture masks with NaN, which ``np.isfinite`` rejects anyway —
    so it never exercised the masked-array path. netCDF4 returns a masked
    array here, and ``np.asarray`` on it exposes the raw -999 as if it were
    data.
    """
    gdir = cache / 'glodap'
    gdir.mkdir(parents=True, exist_ok=True)
    depth = np.array([0.0, 500.0, 2000.0])
    lat = np.arange(-89.5, 90.5, 1.0)
    lon = np.arange(20.5, 380.5, 1.0)
    ds = netCDF4.Dataset(gdir / glodap_local.GLODAP_FILE, 'w')
    ds.createDimension('depth', depth.size)
    ds.createDimension('lat', lat.size)
    ds.createDimension('lon', lon.size)
    ds.createVariable('Depth', 'f8', ('depth',))[:] = depth
    ds.createVariable('lat', 'f8', ('lat',))[:] = lat
    ds.createVariable('lon', 'f8', ('lon',))[:] = lon
    v = ds.createVariable('pHtsinsitutp', 'f4', ('depth', 'lat', 'lon'),
                          fill_value=fill)
    data = np.full((depth.size, lat.size, lon.size), fill, dtype='f4')
    data[0, 120, 299] = 8.10
    data[1, 120, 299] = 8.05
    v[:] = data
    ds.close()


def test_fillvalue_levels_are_dropped_not_read_as_ph(tmp_path, monkeypatch):
    """A ``_FillValue`` level must be trimmed, never returned as pH.

    Reading it as data gives pH = -999, which propagates into
    Francois-Garrison as a wildly wrong boric-acid term.
    """
    root = tmp_path / 'fv_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    _cache.invalidate_grids()
    _write_glodap_fillvalue(root)

    depths, ph = glodap_local.fetch_ph_profile((30.5, -40.5))
    assert np.all(np.isfinite(ph)), f"non-finite pH survived: {ph}"
    assert np.all((ph > 6.0) & (ph < 9.0)), f"implausible pH values: {ph}"
    np.testing.assert_allclose(depths, [0.0, 500.0])
    np.testing.assert_allclose(ph, [8.10, 8.05], rtol=1e-5)
