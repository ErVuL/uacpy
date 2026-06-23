"""Tests for the NSIDC sea-ice climatology backend (uacpy.data.seaice_local)."""

import pickle

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import seaice_local


def test_decode_to_fraction():
    a = np.array([[0, 500, 1000], [2510, 2530, 2540]], dtype=np.uint16)
    f = seaice_local._to_fraction(a)
    assert f[0].tolist() == [0.0, 0.5, 1.0]      # concentration ×10 → fraction
    assert f[1, 0] == 1.0                         # pole hole → perennial ice
    assert np.isnan(f[1, 1]) and np.isnan(f[1, 2])   # coast / land


class _FakeTF:
    def __init__(self, x, y):
        self._x, self._y = x, y

    def transform(self, lon, lat):
        return self._x, self._y


@pytest.fixture
def synthetic_model(monkeypatch):
    g = seaice_local._GRID['N']
    # A point that maps to pixel (row=2, col=3) of the North grid.
    x = g['x0'] + 3 * g['px']
    y = g['y0'] - 2 * g['px']
    north = np.zeros((12, 5, 6), dtype=np.float32)
    north[2, 2, 3] = 0.7                          # March, that pixel
    north[5, 2, 3] = np.nan                       # June, land at that pixel
    model = {'tf': {'N': _FakeTF(x, y), 'S': _FakeTF(0.0, 0.0)},
             'N': north, 'S': np.zeros((12, 5, 6), dtype=np.float32)}
    monkeypatch.setattr(seaice_local, '_model', lambda: model)
    return model


def test_concentration_value(synthetic_model):
    assert seaice_local.fetch_sea_ice_concentration((85.0, 0.0), month=3) == \
        pytest.approx(0.7)


def test_concentration_by_date(synthetic_model):
    assert seaice_local.fetch_sea_ice_concentration((85.0, 0.0),
                                                    date='2026-03-15') == pytest.approx(0.7)


def test_land_raises(synthetic_model):
    with pytest.raises(DataFetchError, match='land / coast'):
        seaice_local.fetch_sea_ice_concentration((85.0, 0.0), month=6)


def test_out_of_grid_is_ice_free(monkeypatch):
    far = {'tf': {'N': _FakeTF(1e9, 1e9), 'S': _FakeTF(1e9, 1e9)},
           'N': np.zeros((12, 5, 6), np.float32), 'S': np.zeros((12, 5, 6), np.float32)}
    monkeypatch.setattr(seaice_local, '_model', lambda: far)
    assert seaice_local.fetch_sea_ice_concentration((85.0, 0.0), month=3) == 0.0


def test_date_and_month_rejected(synthetic_model):
    with pytest.raises(ConfigurationError, match='not both'):
        seaice_local.fetch_sea_ice_concentration((85.0, 0.0), date='2026-03-01', month=3)


def test_requires_date_or_month(synthetic_model):
    with pytest.raises(ConfigurationError, match='date.*month'):
        seaice_local.fetch_sea_ice_concentration((85.0, 0.0))


def test_transect(synthetic_model):
    r, c = seaice_local.fetch_sea_ice_concentration_transect(
        (85.0, 0.0), (85.5, 0.0), month=3, n_points=3)
    assert r.shape == (3,) and c.shape == (3,)


def test_sea_ice_surface_transect(synthetic_model):
    # March pixel reads 0.7 ≥ threshold along the whole transect → an elastic
    # Surface carrier with one node per waypoint.
    from uacpy.core.surface import Surface
    surf = seaice_local.sea_ice_surface_transect(
        (85.0, 0.0), (85.5, 0.0), month=3, n_points=4)
    assert isinstance(surf, Surface)
    assert surf.n_ranges == 4 and surf.is_elastic
    assert surf.at(range=0).acoustic_type == 'half-space'


def test_sea_ice_surface_gates_on_threshold():
    # Below the 15 % ice-edge → open water (no surface override).
    assert seaice_local.sea_ice_surface(0.10) is None
    assert seaice_local.sea_ice_surface(0.30, threshold=0.5) is None
    # At/above → an elastic ice canopy with the COA canonical values.
    bp = seaice_local.sea_ice_surface(0.15)
    assert bp is not None and bp.acoustic_type == 'half-space'
    assert (bp.sound_speed, bp.shear_speed, bp.density) == (3500.0, 1800.0, 0.9)
    assert (bp.attenuation, bp.shear_attenuation) == (0.5, 1.0)


def test_sea_ice_surface_nan_is_open_water():
    # A non-finite concentration (NaN land/coast/out-of-grid cell) must be
    # treated as open water, never silently as ice: NaN < threshold is False,
    # so without the isfinite guard a land pixel would get an elastic canopy.
    assert seaice_local.sea_ice_surface(np.nan) is None
    assert seaice_local.sea_ice_surface(float('inf')) is None


def test_fetch_sea_ice_surface(synthetic_model):
    # 0.7 at the March pixel ≥ threshold → ice; June is land → raises.
    assert seaice_local.fetch_sea_ice_surface((85.0, 0.0), month=3) is not None
    with pytest.raises(DataFetchError, match='land / coast'):
        seaice_local.fetch_sea_ice_surface((85.0, 0.0), month=6)


def test_download_builds_climatology(tmp_path, monkeypatch):
    pytest.importorskip('tifffile')             # default dep; guard a stripped-down env
    monkeypatch.setattr(seaice_local, 'http_get', lambda url, **kw: b'TIFF')
    monkeypatch.setattr('tifffile.imread',
                        lambda b: np.full((4, 4), 600, dtype=np.uint16))
    out = seaice_local.download_seaice_db(cache_dir=str(tmp_path), years=[2023])
    assert out.exists()
    climo = pickle.load(open(out, 'rb'))
    assert climo['N'].shape == (12, 4, 4)
    assert np.allclose(climo['N'], 0.6)          # 600/1000
