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


def test_fetch_point_depth_dispatches_gmrt(monkeypatch):
    monkeypatch.setattr(gmrt_live, 'http_get', _stub_point(-1234))
    assert bathymetry.fetch_point_depth((43.2, 7.5), source='gmrt') == 1234.0


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


@pytest.mark.requires_network
def test_live_gmrt_point():
    try:
        d = gmrt_live.point_depth((43.2, 7.5))
    except DataFetchError as exc:
        pytest.skip(f"GMRT unreachable: {exc.message}")
    assert 1000.0 < d < 4000.0                         # Ligurian Sea, deep
