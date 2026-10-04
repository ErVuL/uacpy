"""Tests for the Diesing 2020 deep-sea lithology backend (uacpy.data.diesing_local).

The class lookup (``_class_code``: reproject + raster index) is mocked, so these
run without the raster or pyproj; the download test mocks the HTTP fetch.
"""

import io
import zipfile

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import _http, diesing_local, fetch_bottom_transect
from uacpy.data.sediment import bottom_from_grain_size


@pytest.mark.parametrize('code, litho, phi', [
    (1, 'calcareous sediment', 7.5),
    (2, 'clay', 9.0),
    (3, 'diatom ooze', 9.0),
    (4, 'lithogenous sediment', 4.0),
    (5, 'radiolarian ooze', 8.0),
])
def test_class_to_lithology(monkeypatch, code, litho, phi):
    monkeypatch.setattr(diesing_local, '_class_code',
                        lambda lat, lon: (code, None))
    sub = diesing_local.fetch_seafloor_lithology((0.0, -150.0))
    assert sub.material == litho and sub.grain_size_phi == phi
    bp = diesing_local.fetch_bottom_diesing((0.0, -150.0))
    assert isinstance(bp, BoundaryProperties)
    assert bp.acoustic_type == 'half-space' and bp.grain_size_phi == phi
    assert [(p.source.id, p.requested_point) for p in bp.data_sources] == [
        ('diesing', (0.0, -150.0))]


def test_no_coverage_raises(monkeypatch):
    monkeypatch.setattr(diesing_local, '_class_code', lambda lat, lon: None)
    with pytest.raises(DataFetchError, match='deeper than 500 m'):
        diesing_local.fetch_seafloor_lithology((56.0, 3.0))         # shelf


def test_transect_classifies_each_waypoint_from_its_own_cell(monkeypatch):
    # Latitude-banded class raster: calcareous sediment (1, ϕ 7.5) south of
    # 0.5°N, radiolarian ooze (5, ϕ 8.0) to 1.5°N, clay (2, ϕ 9.0) beyond.
    # The waypoints at 0°, 1°, 2°N each read their own band.
    def code_at(lat, lon):
        if lat < 0.5:
            return 1, None
        return (5 if lat < 1.5 else 2), None

    monkeypatch.setattr(diesing_local, '_class_code', code_at)
    rdb = fetch_bottom_transect((0.0, 0.0), (2.0, 0.0), source='diesing',
                                n_points=3)
    assert isinstance(rdb, Bottom)
    assert rdb.ranges.shape == (3,) and rdb.ranges[0] == 0.0
    expected = [bottom_from_grain_size(phi).sound_speed
                for phi in (7.5, 8.0, 9.0)]
    assert rdb.halfspace_sound_speed.tolist() == pytest.approx(expected)
    # Hamilton c_p falls with ϕ: 1513.71 > 1506.47 > 1494.90 m/s.
    assert np.all(np.diff(rdb.halfspace_sound_speed) < 0)


def test_download_extracts_raster(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('lithology_classes.tif', b'FAKE-GEOTIFF')
        zf.writestr('README.txt', b'x')
    monkeypatch.setattr(_http, 'curl_download', lambda *a, **kw: False)
    monkeypatch.setattr(_http, 'http_get', lambda url, **kw: buf.getvalue())
    out = diesing_local.download_diesing_db(cache_dir=str(tmp_path))
    assert out.name == 'lithology_classes.tif'
    assert out.read_bytes() == b'FAKE-GEOTIFF'


def test_missing_cache_names_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    diesing_local._model.memo.clear()
    with pytest.raises(ConfigurationError, match='install.sh --data diesing'):
        diesing_local.fetch_bottom_diesing((0.0, -150.0))


class TestTheRasterReadKeepsPillowsDecompressionGuard:
    """``Image.MAX_IMAGE_PIXELS`` is Pillow's process-wide guard against a
    decompression bomb. Reading the raster leaves it as the user set it, and
    reads under it: a raster within the guard loads, one past it is refused."""

    WIDTH, HEIGHT = 20, 10          # 200 pixels

    @pytest.fixture
    def raster_cache(self, tmp_path, monkeypatch):
        pytest.importorskip('pyproj')
        from PIL import Image, TiffImagePlugin, TiffTags
        root = tmp_path / 'data_cache'
        (root / 'diesing').mkdir(parents=True)
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[33550] = (10000.0, 10000.0, 0.0)             # ModelPixelScale
        tags.tagtype[33550] = TiffTags.DOUBLE
        tags[33922] = (0.0, 0.0, 0.0, -1.0e5, 5.0e4, 0.0)  # ModelTiepoint
        tags.tagtype[33922] = TiffTags.DOUBLE
        codes = np.full((self.HEIGHT, self.WIDTH), 2.0, dtype=np.float32)
        Image.fromarray(codes, mode='F').save(
            root / 'diesing' / diesing_local.RASTER_FILE, tiffinfo=tags)
        monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
        return Image

    def test_a_raster_within_the_guard_loads_and_leaves_it_set(
            self, raster_cache, monkeypatch):
        Image = raster_cache
        monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 200)
        model = diesing_local._build_model()
        assert (model['H'], model['W']) == (self.HEIGHT, self.WIDTH)
        assert Image.MAX_IMAGE_PIXELS == 200

    def test_a_raster_past_the_guard_is_refused(self, raster_cache,
                                                 monkeypatch):
        # Pillow raises past twice the limit: 200 pixels > 2 * 99.
        from uacpy.data._cache import UnreadableCacheError
        Image = raster_cache
        monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 99)
        with pytest.raises(UnreadableCacheError,
                           match='DecompressionBombError'):
            diesing_local._build_model()
        assert Image.MAX_IMAGE_PIXELS == 99


def test_the_antimeridian_reads_alike_from_either_side(monkeypatch):
    """Wagner IV sends +180 and −180 to opposite ends of the same parallel, and
    the rasterized nodata margin covers one end without covering the other."""
    pytest.importorskip('pyproj')
    from uacpy.data import diesing_local
    tf = diesing_local._pyproj_transformer()
    sx = 1.0e6
    x_edge = abs(tf.transform(180.0, 0.0)[0])
    width = int(np.ceil(2 * x_edge / sx)) + 4
    arr = np.full((8, width), 2.0, dtype=np.float32)    # class 2 = clay
    arr[:, :4] = -3.4e38                               # west margin: nodata
    monkeypatch.setattr(diesing_local, '_model', lambda: {
        'arr': arr, 'x0': -x_edge - 2 * sx, 'y0': 4 * sx, 'sx': sx, 'sy': sx,
        'tf': tf, 'H': arr.shape[0], 'W': width})
    assert diesing_local._class_code(0.0, 180.0)[0] == 2
    assert diesing_local._class_code(0.0, -180.0)[0] == 2
    # An interior miss is still a miss — the retry is edge-only.
    arr[:] = -3.4e38
    assert diesing_local._class_code(0.0, 0.0) is None


def test_the_diesing_transformer_delegates_to_the_seaice_builder(monkeypatch):
    from uacpy.data import diesing_local, seaice_local
    calls = {}

    def fake_builder(epsg, *, backend='sea-ice'):
        calls['epsg'], calls['backend'] = epsg, backend
        return 'sentinel-transformer'

    monkeypatch.setattr(seaice_local, '_pyproj_transformer', fake_builder)
    assert diesing_local._pyproj_transformer() == 'sentinel-transformer'
    assert calls['epsg'] == diesing_local.WAGNER4_PROJ
    assert 'Diesing' in calls['backend']
