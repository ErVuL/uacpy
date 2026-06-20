"""Tests for the GMRT live bathymetry backend (uacpy.data.gmrt_live).

The PointServer/GridServer HTTP layer is stubbed with canned responses so these
run offline; one ``requires_network`` test hits the live service.
"""

import numpy as np
import pytest

from uacpy.core.exceptions import DataFetchError
from uacpy.data import bathymetry, gmrt_live


def _stub_point(value):
    def fake(url, *, timeout, verbose, source='data', user_agent='uacpy'):
        return str(value).encode()
    return fake


def test_point_depth(monkeypatch):
    monkeypatch.setattr(gmrt_live, 'http_get', _stub_point(-2455))
    assert gmrt_live.point_depth((43.2, 7.5)) == 2455.0


def test_point_land_raises(monkeypatch):
    monkeypatch.setattr(gmrt_live, 'http_get', _stub_point(150))   # +up = land
    with pytest.raises(DataFetchError, match='land'):
        gmrt_live.point_depth((45.0, 7.0))


def test_unparseable_raises(monkeypatch):
    monkeypatch.setattr(gmrt_live, 'http_get', _stub_point('n/a'))
    with pytest.raises(DataFetchError, match='unparseable'):
        gmrt_live.point_depth((43.2, 7.5))


def test_depths_along(monkeypatch):
    monkeypatch.setattr(gmrt_live, 'http_get', _stub_point(-1000))
    out = gmrt_live.depths_along([43.0, 43.1, 43.2], [7.0, 7.1, 7.2])
    assert out.tolist() == [1000.0, 1000.0, 1000.0]


def test_fetch_bathy_dispatches_gmrt(monkeypatch):
    monkeypatch.setattr(gmrt_live, 'http_get', _stub_point(-1234))
    assert bathymetry.fetch_bathy((43.2, 7.5), source='gmrt') == 1234.0


def test_region_grid(monkeypatch, tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    # Build a synthetic COARDS grid (what GridServer?format=coards returns).
    path = tmp_path / 'grid.nc'
    lon = np.linspace(7.0, 7.5, 12)
    lat = np.linspace(43.0, 43.5, 10)
    alt = np.full((lat.size, lon.size), -2000.0)
    alt[0, 0] = 30.0                                   # one land node
    ds = netCDF4.Dataset(path, 'w')
    ds.createDimension('lon', lon.size); ds.createDimension('lat', lat.size)
    ds.createVariable('lon', 'f8', ('lon',))[:] = lon
    ds.createVariable('lat', 'f8', ('lat',))[:] = lat
    ds.createVariable('altitude', 'f4', ('lat', 'lon'))[:] = alt
    ds.close()
    blob = path.read_bytes()

    def fake(url, *, timeout, verbose, source='data', user_agent='uacpy'):
        return blob
    monkeypatch.setattr(gmrt_live, 'http_get', fake)

    lats, lons, depth = gmrt_live.region_grid((43.0, 43.5), (7.0, 7.5), 6, 6)
    assert depth.shape == (6, 6)
    assert np.nanmin(depth) == 2000.0
    assert np.isnan(depth).sum() >= 1                  # the land node


def test_nearest_indices_ascending_and_descending():
    # Nearest-node selection must be exact on an ascending axis (not the
    # upper-biased searchsorted insertion index) and equally correct on a
    # descending axis (GMRT COARDS latitude is commonly stored descending).
    asc = np.array([10.0, 11.0, 12.0, 13.0])
    desc = asc[::-1].copy()
    q = np.array([11.4, 12.6])                     # nearest nodes: 11.0, 13.0
    ia = gmrt_live._nearest_indices(asc, q)
    idesc = gmrt_live._nearest_indices(desc, q)
    assert asc[ia].tolist() == [11.0, 13.0]
    assert desc[idesc].tolist() == [11.0, 13.0]


def test_region_grid_descending_latitude(monkeypatch, tmp_path):
    # A descending-latitude COARDS subset must still sample the right cells —
    # the old searchsorted resample returned meaningless indices on a
    # descending axis. Encode depth as the latitude so a wrong row is visible.
    netCDF4 = pytest.importorskip('netCDF4')
    path = tmp_path / 'grid_desc.nc'
    lon = np.linspace(7.0, 7.5, 6)
    lat = np.linspace(43.5, 43.0, 10)              # DESCENDING
    # altitude[r, c] = -(100 + lat_index*10): each row a distinct depth.
    alt = -(100.0 + np.arange(lat.size)[:, None] * 10.0
            + np.zeros((lat.size, lon.size)))
    ds = netCDF4.Dataset(path, 'w')
    ds.createDimension('lon', lon.size); ds.createDimension('lat', lat.size)
    ds.createVariable('lon', 'f8', ('lon',))[:] = lon
    ds.createVariable('lat', 'f8', ('lat',))[:] = lat
    ds.createVariable('altitude', 'f4', ('lat', 'lon'))[:] = alt
    ds.close()
    blob = path.read_bytes()
    monkeypatch.setattr(gmrt_live, 'http_get',
                        lambda url, **kw: blob)

    lats, lons, depth = gmrt_live.region_grid((43.0, 43.5), (7.0, 7.5), 5, 4)
    assert depth.shape == (5, 4)
    # Requested lats ascend; the deepest native row is lat=43.0 (last, index 9),
    # the shallowest is lat=43.5 (index 0). So the smallest requested lat must
    # map to the deepest cell and the largest to the shallowest.
    assert depth[0, 0] > depth[-1, 0]              # 43.0 deeper than 43.5
    assert np.nanmax(depth) == 100.0 + 9 * 10.0    # 190 m at lat 43.0
    assert np.nanmin(depth) == 100.0               # 100 m at lat 43.5


@pytest.mark.requires_network
def test_live_gmrt_point():
    try:
        d = gmrt_live.point_depth((43.2, 7.5))
    except DataFetchError as exc:
        pytest.skip(f"GMRT unreachable: {exc.message}")
    assert 1000.0 < d < 4000.0                         # Ligurian Sea, deep
