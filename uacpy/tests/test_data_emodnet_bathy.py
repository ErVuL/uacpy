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


@pytest.mark.parametrize('base_url, root', [
    (None, 'https://erddap.emodnet.eu/erddap/griddap/'),
    ('https://dtm.example.org/erddap/griddap',
     'https://dtm.example.org/erddap/griddap/')])
def test_the_dtm_requests_go_to_base_url(monkeypatch, base_url, root):
    """``fetch_bathy(source='emodnet_dtm', base_url=)`` reads every tile
    under the griddap root given (point, transect and grid); ``None`` reads
    the module constant."""
    urls = []

    def fake(url, *, timeout, verbose, source='data', user_agent='uacpy'):
        urls.append(url)
        return _POINT_CSV.format(elev=-37.4).encode()
    monkeypatch.setattr(emo, 'http_get', fake)
    assert bathymetry.fetch_bathy((50.0, 0.0), source='emodnet_dtm',
                                  base_url=base_url) == pytest.approx(37.4)
    bathymetry.fetch_bathy_transect((50.0, 0.0), (50.1, 0.1), n_points=2,
                                    source='emodnet_dtm', base_url=base_url)
    bathymetry.fetch_bathy_grid((50.0, 50.1), (0.0, 0.1), n_lat=2, n_lon=2,
                                source='emodnet_dtm', base_url=base_url)
    assert len(urls) == 4
    assert all(u.startswith(root + emo.DATASETS[0] + '.csv?') for u in urls)


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


def test_a_nan_cell_raises_naming_the_valueless_cell(monkeypatch):
    monkeypatch.setattr(emo, 'http_get', _stub_point('NaN'))
    with pytest.raises(DataFetchError, match='no value at the nearest cell'):
        emo.point_depth((50.0, 0.0))


def test_depths_along(monkeypatch):
    monkeypatch.setattr(emo, 'http_get', _stub_point(-1000.0))
    out = emo.depths_along([50.0, 50.1], [0.0, 0.1])
    assert out.tolist() == [1000.0, 1000.0]


def test_the_node_the_reply_names_is_returned_with_the_depth(monkeypatch):
    monkeypatch.setattr(emo, 'http_get', _stub_point(-37.4))
    depth, node = emo._depth_and_node((50.0, 0.0), timeout=5.0, verbose=False)
    assert depth == pytest.approx(37.4)
    assert node == pytest.approx((50.000520833, 0.000520833), abs=1e-12)


@pytest.mark.parametrize('transect_to', [None, (50.1, 0.1)],
                         ids=['point', 'transect'])
def test_an_environment_records_the_dtm_node_read(monkeypatch, transect_to):
    """The ERDDAP reply names the DTM node it read, so the bathymetry
    record carries it as the data point (57.9 m north, 37.2 m east: 0.069 km
    off, inside its 1/16' cell: no notice), not the requested point at
    0 km."""
    from uacpy.core.exceptions import ProvenanceWarning
    from uacpy.tests.conftest import recorded_warnings
    monkeypatch.setattr(emo, 'http_get', _stub_point(-88.0))
    with recorded_warnings() as caught:
        env = data.fetch_environment((50.0, 0.0), transect_to=transect_to,
                                     bathymetry_sources='emodnet_dtm',
                                     ssp=1500.0, bottom=2.0, n_points=3)
    (record,) = env.bathymetry.data_sources
    assert record.source.id == 'emodnet_dtm'
    assert record.data_point == pytest.approx((50.000520833, 0.000520833), abs=1e-12)
    assert record.requested_point == (50.0, 0.0)
    assert (record.point_kind, record.cell_size_deg) == ('cell', 1.0 / 960.0)
    assert record.offset_km == pytest.approx(0.0688, abs=2e-4)
    assert not [w for w in caught if issubclass(w.category, ProvenanceWarning)
                and 'EMODnet' in str(w.message)]


def test_fetch_bathy_dispatches_emodnet(monkeypatch):
    monkeypatch.setattr(emo, 'http_get', _stub_point(-88.0))
    assert bathymetry.fetch_bathy((50.0, 0.0), source='emodnet_dtm') == pytest.approx(88.0)


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


def test_a_region_failure_names_what_each_tile_answered(monkeypatch):
    def fake(url, *, timeout, verbose, source='data', user_agent='uacpy'):
        raise DataFetchError("Could not reach the host: timed out")
    monkeypatch.setattr(emo, 'http_get', fake)
    with pytest.raises(DataFetchError, match='timed out') as info:
        emo.region_grid((43.0, 43.5), (7.0, 7.5), 4, 4)
    assert all(d in info.value.message for d in emo.DATASETS)


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
        d = emo.point_depth((51.5, 2.5))               # southern North Sea,
        #   ~50 km off both coasts; the live DTM cell holds ~34 m of water
    except DataFetchError as exc:
        pytest.skip(f"EMODnet DTM unreachable: {exc.message}")
    assert 0.0 < d < 200.0
