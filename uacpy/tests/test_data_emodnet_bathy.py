"""Tests for the EMODnet DTM live bathymetry backend (emodnet_bathy_live).

The ERDDAP griddap HTTP layer is stubbed with canned CSV responses so these run
offline; one ``requires_network`` test hits the live service.
"""

import numpy as np
import pytest

import uacpy.data as data
from uacpy.core.exceptions import DataFetchError
from uacpy.data import bathymetry, emodnet_bathy_live as emo

_POINT_CSV = ("latitude,longitude,elevation\n"
              "degrees_north,degrees_east,m\n"
              "50.000520833,0.000520833,{elev}\n")


def _stub_point(elev):
    def fake(url, *, timeout, verbose, source='data', user_agent='uacpy'):
        return _POINT_CSV.format(elev=elev).encode()
    return fake


def test_point_depth(monkeypatch):
    monkeypatch.setattr(emo, 'http_get', _stub_point(-37.4))
    assert emo.point_depth((50.0, 0.0)) == pytest.approx(37.4)


def test_point_land_raises(monkeypatch):
    monkeypatch.setattr(emo, 'http_get', _stub_point(12.0))    # +up = land
    with pytest.raises(DataFetchError, match='land'):
        emo.point_depth((50.0, 0.0))


def test_out_of_coverage_raises(monkeypatch):
    # Both tiles report out-of-range (HTTP error surfaced as DataFetchError).
    def fake(url, *, timeout, verbose, source='data', user_agent='uacpy'):
        raise DataFetchError("HTTP 404: outside range")
    monkeypatch.setattr(emo, 'http_get', fake)
    with pytest.raises(DataFetchError, match='no coverage'):
        emo.point_depth((-30.0, 150.0))


def test_caribbean_tile_fallback(monkeypatch):
    # The European tile 404s; the Caribbean tile answers → depth resolved.
    def fake(url, *, timeout, verbose, source='data', user_agent='uacpy'):
        if 'carib' in url:
            return _POINT_CSV.format(elev=-1500.0).encode()
        raise DataFetchError("HTTP 404: outside range")
    monkeypatch.setattr(emo, 'http_get', fake)
    assert emo.point_depth((15.0, -70.0)) == pytest.approx(1500.0)


def test_nan_cell_is_no_coverage(monkeypatch):
    monkeypatch.setattr(emo, 'http_get', _stub_point('NaN'))
    with pytest.raises(DataFetchError, match='no coverage'):
        emo.point_depth((50.0, 0.0))


def test_depths_along(monkeypatch):
    monkeypatch.setattr(emo, 'http_get', _stub_point(-1000.0))
    out = emo.depths_along([50.0, 50.1], [0.0, 0.1])
    assert out.tolist() == [1000.0, 1000.0]


def test_fetch_bathy_dispatches_emodnet(monkeypatch):
    monkeypatch.setattr(emo, 'http_get', _stub_point(-88.0))
    assert bathymetry.fetch_bathy((50.0, 0.0), source='emodnet') == pytest.approx(88.0)


def test_region_grid(monkeypatch):
    grid_csv = ("latitude,longitude,elevation\n"
                "degrees_north,degrees_east,m\n"
                "43.0,7.0,-2000\n"
                "43.0,7.5,-2000\n"
                "43.5,7.0,-2000\n"
                "43.5,7.5,30\n")                       # one land node
    monkeypatch.setattr(emo, 'http_get',
                        lambda url, **kw: grid_csv.encode())
    lats, lons, depth = emo.region_grid((43.0, 43.5), (7.0, 7.5), 4, 4)
    assert depth.shape == (4, 4)
    assert np.nanmin(depth) == 2000.0
    assert np.isnan(depth).sum() >= 1                  # the land node


def test_fetch_environment_records_emodnet_provenance(monkeypatch):
    # A literal SSP keeps the run offline (no SSP fetch); the bathy source is
    # EMODnet DTM, whose provenance id must be the distinct 'emodnet_dtm' entry
    # (not the EMODnet Geology 'emodnet' substrate source).
    monkeypatch.setattr(emo, 'http_get', _stub_point(-120.0))
    env = data.fetch_environment((50.0, 0.0), bathymetry_sources='emodnet_dtm',
                                 ssp=1500.0)
    assert env.depth == pytest.approx(120.0)
    assert [s.source.id for s in env.data_sources] == ['emodnet_dtm']


@pytest.mark.requires_network
def test_live_emodnet_point():
    try:
        d = emo.point_depth((51.0, 2.5))               # southern North Sea
    except DataFetchError as exc:
        pytest.skip(f"EMODnet DTM unreachable: {exc.message}")
    assert 0.0 < d < 200.0
