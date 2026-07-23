"""Tests for the offline NRL/Graw seabed bulk-density backend (uacpy.data.graw_local).

Builds a synthetic ``$UACPY_DATA_CACHE`` Graw NetCDF (tiny ``z(lat, lon)``
density grid, mirroring the GlobSed schema) so the density reader, the
density → grain-size inversion and a ``bottom_sources='graw'``
``fetch_environment`` run from the cache with no network. Skipped where netCDF4
is unavailable; one ``requires_network`` test hits the live Zenodo grid.
"""

import numpy as np
import pytest

netCDF4 = pytest.importorskip('netCDF4')

import uacpy.data as data
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import gebco_local, graw_local, woa23_local


def _write_graw(cache, *, value=1.962):
    """Synthetic Graw density grid: one 1° global field, ``value`` everywhere
    except a NaN cell at (0.5, 0.5) to exercise the no-data guard."""
    gdir = cache / 'graw'; gdir.mkdir(parents=True)
    lat = np.arange(-89.5, 90.5, 1.0)
    lon = np.arange(-179.5, 180.5, 1.0)
    z = np.full((lat.size, lon.size), value, dtype=np.float32)
    z[90, 180] = np.nan                                # (0.5, 0.5)
    ds = netCDF4.Dataset(gdir / graw_local.GRAW_FILE, 'w')
    ds.createDimension('lat', lat.size)
    ds.createDimension('lon', lon.size)
    ds.createVariable('lat', 'f8', ('lat',))[:] = lat
    ds.createVariable('lon', 'f8', ('lon',))[:] = lon
    ds.createVariable('z', 'f4', ('lat', 'lon'))[:] = z
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


_FILL = 9.96921e36


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


@pytest.fixture
def graw_cache(tmp_path, monkeypatch):
    """Cache holding only the synthetic Graw density grid."""
    root = tmp_path / 'graw_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    graw_local._GRID.clear()
    _write_graw(root)
    return root


@pytest.fixture
def full_cache(tmp_path, monkeypatch):
    """GEBCO + WOA23 + Graw — a fully offline bottom_sources='graw' run."""
    root = tmp_path / 'full_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    gebco_local._GRID.clear()
    woa23_local._DATASETS.clear()
    graw_local._GRID.clear()
    _write_gebco(root)
    _write_woa(root)
    _write_graw(root)
    return root


# ── density reader ──────────────────────────────────────────────────────────

def test_density_at_point(graw_cache):
    assert graw_local.fetch_seabed_density((30.5, -40.5)) == pytest.approx(
        1.962, abs=1e-3)


def test_density_no_data_raises(graw_cache):
    with pytest.raises(DataFetchError, match='no seabed density'):
        graw_local.fetch_seabed_density((0.5, 0.5))


def test_density_transect(graw_cache):
    ranges, rho = graw_local.fetch_seabed_density_transect(
        (30.5, -40.5), (32.5, -40.5), n_points=4)
    assert ranges.shape == (4,) and ranges[0] == 0.0
    assert np.allclose(rho, 1.962, atol=1e-3)


def test_missing_cache_names_install_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    graw_local._GRID.clear()
    with pytest.raises(ConfigurationError, match='install.sh --data graw'):
        graw_local.fetch_seabed_density((30.5, -40.5))


# ── density → bottom ────────────────────────────────────────────────────────

def test_bottom_uses_measured_density(graw_cache):
    # 1.962 g/cm³ is exactly Hamilton's fine sand row (ϕ = 2.5, ratio 1.152):
    # the bottom carries the *measured* density and the ϕ-derived speed.
    bp = graw_local.fetch_bottom_graw((30.5, -40.5))
    assert bp.acoustic_type == 'half-space'
    assert bp.density == pytest.approx(1.962, abs=1e-3)
    assert bp.grain_size_phi == pytest.approx(2.5, abs=0.05)
    assert bp.sound_speed == pytest.approx(1.152 * 1510.0, rel=0.01)
    assert bp.attenuation > 0.0


def test_bottom_scales_to_water_sound_speed(graw_cache):
    ref = graw_local.fetch_bottom_graw((30.5, -40.5))
    warm = graw_local.fetch_bottom_graw((30.5, -40.5), water_sound_speed=1540.0)
    assert warm.sound_speed == pytest.approx(ref.sound_speed * 1540.0 / 1510.0,
                                             rel=1e-6)
    assert warm.density == ref.density                 # measured, not scaled


def test_density_clamped_to_table(tmp_path, monkeypatch):
    # Densities outside the Hamilton table clamp to its end members instead of
    # extrapolating to unphysical grain sizes.
    root = tmp_path / 'clamp_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    graw_local._GRID.clear()
    _write_graw(root, value=2.4)                       # denser than coarse sand
    bp = graw_local.fetch_bottom_graw((30.5, -40.5))
    assert bp.grain_size_phi == pytest.approx(0.5)     # coarse-sand end member
    assert bp.density == pytest.approx(2.4, abs=1e-3)


def test_bottom_transect(graw_cache):
    bottom = graw_local.fetch_bottom_graw_transect(
        (30.5, -40.5), (32.5, -40.5), n_points=4)
    assert np.allclose(bottom.halfspace_density, 1.962, atol=1e-3)


# ── fetch_environment integration ───────────────────────────────────────────

def test_fetch_environment_bottom_sources_graw(full_cache):
    env = data.fetch_environment((30.5, -40.5), ssp_sources='local',
                                 bottom_sources='graw')
    assert env.bottom.halfspace_density[0] == pytest.approx(1.962, abs=1e-3)
    assert 'graw' in [s.source.id for s in env.data_sources]


# ── live ────────────────────────────────────────────────────────────────────

@pytest.mark.requires_network
def test_live_graw_download(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'live'))
    graw_local._GRID.clear()
    try:
        graw_local.download_graw_db(timeout=300.0)
    except DataFetchError as exc:
        pytest.skip(f"Zenodo unreachable: {exc.message}")
    rho = graw_local.fetch_seabed_density((30.0, -40.0))  # mid-Atlantic
    assert 1.0 < rho < 2.3
