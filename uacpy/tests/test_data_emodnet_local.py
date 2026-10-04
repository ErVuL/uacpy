"""Tests for the offline EMODnet seabed backend (uacpy.data.emodnet_local).

Mocks the WFS so the download/index/query path runs without network. Skipped
where shapely (a default uacpy dependency) is unavailable.
"""

import json

import numpy as np
import pytest

pytest.importorskip('shapely')

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import emodnet_local, fetch_bottom_transect


def _poly_feature(folk, box):
    minx, miny, maxx, maxy = box           # lon0, lat0, lon1, lat1
    ring = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
    return {'type': 'Feature', 'properties': {'folk_5cl': folk},
            'geometry': {'type': 'Polygon', 'coordinates': [ring]}}


def _fc(features):
    return json.dumps({'type': 'FeatureCollection', 'features': features})


def test_download_builds_wkb_index(tmp_path, monkeypatch):
    feats = [_poly_feature(2, (-41, 30, -40, 31)),
             _poly_feature(5, (2, 50, 3, 51))]
    monkeypatch.setattr(emodnet_local, 'http_get', lambda url, **kw: _fc(feats))
    out = emodnet_local.download_emodnet_db(cache_dir=str(tmp_path))
    assert out.exists()
    with np.load(out, allow_pickle=False) as z:
        assert z['codes'].tolist() == [2, 5]
        # offsets are n+1 cut points into the concatenated WKB blob
        assert z['offsets'].tolist() == [0, z['offsets'][1], z['wkb'].size]


def test_query_after_download(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path))
    emodnet_local._index.memo.clear()
    feats = [_poly_feature(2, (-41, 30, -40, 31))]      # Sand
    monkeypatch.setattr(emodnet_local, 'http_get', lambda url, **kw: _fc(feats))
    emodnet_local.download_emodnet_db()                 # → <cache>/emodnet/

    bp = emodnet_local.fetch_bottom_emodnet_local((30.5, -40.5))
    assert bp.acoustic_type == 'half-space' and bp.grain_size_phi == 2.0
    with pytest.raises(DataFetchError, match='European seas only'):
        emodnet_local.fetch_emodnet_substrate_local((0.0, -140.0))   # mid-Pacific, no polygon


def test_transect_holds_and_varies(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path))
    emodnet_local._index.memo.clear()
    feats = [_poly_feature(2, (-41, 30, -40, 31)),
             _poly_feature(1, (-41, 31, -40, 32))]      # Sand then Mud band
    monkeypatch.setattr(emodnet_local, 'http_get', lambda url, **kw: _fc(feats))
    emodnet_local.download_emodnet_db()
    rdb = fetch_bottom_transect(
        (30.5, -40.5), (31.5, -40.5), source='emodnet', n_points=3)
    assert rdb.halfspace_sound_speed.shape == (3,)
    assert (rdb.halfspace_sound_speed > 1500).all()
    # Sand (ϕ 2.0) at the start, Mud to muddy Sand (ϕ 5.0) at the end: the
    # first and last sound speeds differ, sand the faster.
    assert rdb.halfspace_sound_speed[0] != rdb.halfspace_sound_speed[-1]
    assert rdb.halfspace_sound_speed[0] > rdb.halfspace_sound_speed[-1]


def test_unknown_folk_class_refuses_default(tmp_path, monkeypatch):
    # An out-of-range Folk-5 code must raise rather than fabricate a default
    # 'mixed sediment' bottom.
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path))
    emodnet_local._index.memo.clear()
    feats = [_poly_feature(99, (-41, 30, -40, 31))]     # bogus class
    monkeypatch.setattr(emodnet_local, 'http_get', lambda url, **kw: _fc(feats))
    emodnet_local.download_emodnet_db()
    with pytest.raises(DataFetchError, match='unrecognised Folk-5'):
        emodnet_local.fetch_bottom_emodnet_local((30.5, -40.5))


def test_missing_cache_names_install_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    emodnet_local._index.memo.clear()
    with pytest.raises(ConfigurationError, match='install.sh --data emodnet'):
        emodnet_local.fetch_bottom_emodnet_local((56.0, 3.0))


def test_a_shared_boundary_point_resolves_to_the_lowest_polygon_index(
        tmp_path, monkeypatch):
    pytest.importorskip('shapely')
    from uacpy.data import emodnet_local
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path))
    emodnet_local._index.memo.clear()
    feats = [_poly_feature(2, (-41, 30, -40, 31)),
             _poly_feature(1, (-41, 31, -40, 32))]   # share the lat=31 edge
    fc = json.dumps({'type': 'FeatureCollection', 'features': feats})
    monkeypatch.setattr(emodnet_local, 'http_get', lambda url, **kw: fc)
    emodnet_local.download_emodnet_db()
    sub = emodnet_local.fetch_emodnet_substrate_local((31.0, -40.5))  # on the shared edge
    assert sub.folk_class == 2                            # polygon index 0
    assert sub.folk_class_scheme == 'folk5'


def test_the_package_level_bottom_names_are_the_source_dispatchers():
    """``uacpy.data`` exports one seabed fetcher per shape, taking the source
    as an argument; each provider's own fetchers live in its module only, so
    the offline EMODnet polygons and the NCEI grain-size samples are never
    told apart by a name suffix."""
    from uacpy import data
    from uacpy.data import environment, sediment_db
    assert data.fetch_bottom is environment.fetch_bottom
    assert data.fetch_bottom_transect is environment.fetch_bottom_transect
    for name in ('fetch_bottom_emodnet_local', 'fetch_bottom_grainsize'):
        assert name not in data.__all__
        assert not hasattr(data, name), name
    assert callable(emodnet_local.fetch_bottom_emodnet_local)
    assert callable(sediment_db.fetch_bottom_grainsize)
    assert not hasattr(emodnet_local, 'fetch_bottom_grainsize')
