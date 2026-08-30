"""Tests for the offline coastline cache (uacpy.visualization.basemap).

Natural Earth land polygons are read from the install-time cache first (no
network), with the live GeoJSON as a fallback.
"""

import json
import warnings
from pathlib import Path

import pytest

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


def _corrupt_cache(tmp_path, monkeypatch, resolution='50m'):
    """A present-but-unparseable cache file, and the path it lives at."""
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path))
    cdir = tmp_path / 'coastline'
    cdir.mkdir(parents=True)
    bad = cdir / f'ne_{resolution}_land.geojson'
    bad.write_text('{"type": "FeatureColl')          # truncated mid-write
    return bad


def test_a_corrupt_cache_is_named_before_the_live_source_is_read(tmp_path,
                                                                 monkeypatch):
    """A cache the reader cannot parse is a different state from an absent one
    and takes a different remedy, so it is reported by path rather than
    discarded in silence."""
    bad = _corrupt_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(basemap, 'http_get',
                        lambda url, **kw: json.dumps(_LAND).encode())
    with pytest.warns(UserWarning, match='present but unreadable') as rec:
        rings = basemap.land_polygons('50m')
    assert rings is not None                          # the live source served it
    assert str(bad) in str(rec[0].message)


def test_a_corrupt_cache_with_no_network_is_not_reported_as_a_missing_one(
        tmp_path, monkeypatch):
    """Telling this user to cache the dataset offline names the state they are
    already in; the remedy that moves them on is deleting the damaged file."""
    bad = _corrupt_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(basemap, 'http_get',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('down')))
    with pytest.warns(UserWarning) as rec:
        assert basemap.land_polygons('50m') is None
    unavailable = [str(w.message) for w in rec
                   if 'backdrop unavailable' in str(w.message)]
    assert len(unavailable) == 1
    assert str(bad) in unavailable[0]
    assert 'delete that file' in unavailable[0]


def test_an_intact_cache_is_read_without_a_warning(tmp_path, monkeypatch):
    """The corrupt-cache warning fires on a damaged file only — an ordinary
    read must not train the reader to ignore it."""
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path))
    cdir = tmp_path / 'coastline'
    cdir.mkdir(parents=True)
    (cdir / 'ne_50m_land.geojson').write_text(json.dumps(_LAND))
    monkeypatch.setattr(basemap, 'http_get', _no_network)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        assert basemap.land_polygons('50m') is not None


def test_an_interrupted_download_leaves_no_truncated_cache_file(tmp_path,
                                                                monkeypatch):
    """A writer that opens the destination directly leaves a half-written file
    that every later existence check accepts, and the parse-before-write guard
    upstream cannot see a transfer that dies mid-write."""
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path))
    blob = json.dumps(_LAND).encode()
    monkeypatch.setattr(basemap, 'http_get', lambda url, **kw: blob)
    real_write_bytes = Path.write_bytes

    def dies_half_way(self, data):
        real_write_bytes(self, data[:len(data) // 2])
        raise OSError('no space left on device')

    monkeypatch.setattr(Path, 'write_bytes', dies_half_way)
    with pytest.raises(OSError):
        basemap.download_coastline(resolutions=('110m',))
    dest = tmp_path / 'coastline'
    assert not (dest / 'ne_110m_land.geojson').exists()
    assert list(dest.iterdir()) == []
