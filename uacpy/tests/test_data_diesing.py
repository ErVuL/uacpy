"""Tests for the Diesing 2020 deep-sea lithology backend (uacpy.data.diesing_local).

The class lookup (``_class_code``: reproject + raster index) is mocked, so these
run without the raster or pyproj; the download test mocks the HTTP fetch.
"""

import io
import zipfile

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties, RangeDependentBottom
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import diesing_local


@pytest.mark.parametrize('code, litho, phi', [
    (1, 'calcareous sediment', 7.5),
    (2, 'clay', 9.0),
    (3, 'diatom ooze', 9.0),
    (4, 'lithogenous sediment', 4.0),
    (5, 'radiolarian ooze', 8.0),
])
def test_class_to_lithology(monkeypatch, code, litho, phi):
    monkeypatch.setattr(diesing_local, '_class_code', lambda lat, lon: code)
    sub = diesing_local.fetch_seafloor_lithology((0.0, -150.0))
    assert sub['lithology'] == litho and sub['grain_size_phi'] == phi
    bp = diesing_local.fetch_bottom_diesing((0.0, -150.0))
    assert isinstance(bp, BoundaryProperties)
    assert bp.acoustic_type == 'half-space' and bp.grain_size_phi == phi


def test_no_coverage_raises(monkeypatch):
    monkeypatch.setattr(diesing_local, '_class_code', lambda lat, lon: None)
    with pytest.raises(DataFetchError, match='deeper than 500 m'):
        diesing_local.fetch_seafloor_lithology((56.0, 3.0))         # shelf


def test_transect(monkeypatch):
    codes = iter([1, 2, 3])
    monkeypatch.setattr(diesing_local, '_class_code', lambda lat, lon: next(codes))
    rdb = diesing_local.fetch_bottom_diesing_transect((0.0, 0.0), (2.0, 0.0),
                                                      n_points=3)
    assert isinstance(rdb, RangeDependentBottom)
    assert rdb.sound_speed.shape == (3,) and np.all(np.isfinite(rdb.sound_speed))


def test_download_extracts_raster(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('lithology_classes.tif', b'FAKE-GEOTIFF')
        zf.writestr('README.txt', b'x')
    monkeypatch.setattr(diesing_local, 'http_get', lambda url, **kw: buf.getvalue())
    out = diesing_local.download_diesing_db(cache_dir=str(tmp_path))
    assert out.name == 'lithology_classes.tif'
    assert out.read_bytes() == b'FAKE-GEOTIFF'


def test_missing_cache_names_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    diesing_local._MODEL.clear()
    with pytest.raises(ConfigurationError, match='install.sh --data diesing'):
        diesing_local.fetch_bottom_diesing((0.0, -150.0))
