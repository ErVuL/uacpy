"""Tests for the offline coastline cache (uacpy.visualization.basemap).

Natural Earth land polygons are read from the install-time cache first (no
network), with the live GeoJSON as a fallback.
"""

import json

from uacpy.visualization import basemap

_LAND = {'type': 'FeatureCollection', 'features': [
    {'type': 'Feature',
     'geometry': {'type': 'Polygon',
                  'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}]}


def _no_network(*a, **k):
    raise AssertionError("network used despite a populated coastline cache")


def test_land_polygons_prefers_local_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path))
    cdir = tmp_path / 'coastline'; cdir.mkdir(parents=True)
    (cdir / 'ne_50m_land.geojson').write_text(json.dumps(_LAND))
    monkeypatch.setattr(basemap, 'http_get', _no_network)
    rings = basemap.land_polygons('50m')
    assert rings is not None and rings[0].shape[1] == 2


def test_download_coastline_then_offline_read(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path))
    monkeypatch.setattr(basemap, 'http_get',
                        lambda url, **kw: json.dumps(_LAND).encode())
    out = basemap.download_coastline(resolutions=('110m', '50m'))
    assert len(out) == 2 and all(p.exists() for p in out)
    # subsequent reads come from the cache, not the network
    monkeypatch.setattr(basemap, 'http_get', _no_network)
    assert basemap.land_polygons('110m') is not None


def test_unreachable_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    monkeypatch.setattr(basemap, 'http_get',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('down')))
    assert basemap.land_polygons('50m') is None
