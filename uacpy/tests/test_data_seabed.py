"""Tests for the EMODnet seabed-substrate fetch (uacpy.data.seabed)."""

import json

import pytest

from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import DataFetchError
from uacpy.data import seabed


def _geojson(folk_5cl, txt):
    return json.dumps({'features': [{'properties': {
        'folk_5cl': folk_5cl, 'folk_5cl_txt': txt,
        'original_grain_size': 'Folk - 16'}}]}).encode()


def test_web_mercator_transform():
    x0, y0 = seabed._to_web_mercator(0.0, 0.0)
    assert x0 == pytest.approx(0.0, abs=1e-3)
    assert y0 == pytest.approx(0.0, abs=1e-3)
    x, _ = seabed._to_web_mercator(0.0, 90.0)
    assert x == pytest.approx(10018754.17, rel=1e-6)   # quarter of the world
    # Longitude is normalized: 220 (== -140) maps the same as -140.
    assert seabed._to_web_mercator(0.0, 220.0) == seabed._to_web_mercator(0.0, -140.0)


def test_fetch_seabed_substrate_parses(monkeypatch):
    monkeypatch.setattr(seabed, 'http_get',
                        lambda url, **kw: _geojson(2, '2. Sand'))
    sub = seabed.fetch_seabed_substrate((56.0, 3.0))
    assert sub['folk_5cl'] == 2
    assert sub['folk_5cl_txt'] == '2. Sand'
    assert 'EMODnet' in sub['source']


# The coarse Folk class (ϕ=−1) is outside Hamilton's continental-terrace range,
# so the geoacoustic conversion clamps and warns by design — capture it here.
@pytest.mark.filterwarnings('ignore:grain_size_to_geoacoustics')
@pytest.mark.parametrize('folk,phi', [
    (1, 5.0),    # Mud to muddy Sand
    (2, 2.0),    # Sand
    (3, -1.0),   # Coarse-grained
    (4, 3.0),    # Mixed
])
def test_fetch_bottom_grain_size_classes(monkeypatch, folk, phi):
    monkeypatch.setattr(seabed, 'http_get',
                        lambda url, **kw: _geojson(folk, f'{folk}. x'))
    bp = seabed.fetch_bottom((56.0, 3.0))
    assert isinstance(bp, BoundaryProperties)
    assert bp.acoustic_type == 'half-space'   # universal, model-agnostic
    assert bp.grain_size_phi == phi           # ϕ retained as metadata


def test_fetch_bottom_hard_substrate(monkeypatch):
    monkeypatch.setattr(seabed, 'http_get',
                        lambda url, **kw: _geojson(5, '5. Rock'))
    bp = seabed.fetch_bottom((56.0, 3.0))
    assert bp.acoustic_type == 'half-space'      # limestone preset, not grain-size


def test_no_coverage_raises(monkeypatch):
    monkeypatch.setattr(seabed, 'http_get',
                        lambda url, **kw: json.dumps({'features': []}).encode())
    with pytest.raises(DataFetchError, match='European seas only'):
        seabed.fetch_bottom((0.0, -140.0))         # mid-Pacific: no EMODnet data


def test_fetch_bottom_transect(monkeypatch):
    from uacpy.core.environment import Bottom
    monkeypatch.setattr(seabed, 'http_get',
                        lambda url, **kw: _geojson(2, '2. Sand'))
    rdb = seabed.fetch_bottom_transect((0.0, 0.0), (1.0, 0.0), n_points=4)
    assert isinstance(rdb, Bottom)
    assert rdb.ranges.shape == (4,) and rdb.halfspace_sound_speed.shape == (4,)
    assert rdb.ranges[0] == 0.0
    assert (rdb.halfspace_sound_speed > 1500).all()


def test_fetch_bottom_transect_holds_gaps(monkeypatch):
    # First point covered (Sand), rest uncovered → forward-filled, no raise.
    calls = {'n': 0}

    def flaky(url, **kw):
        calls['n'] += 1
        if calls['n'] == 1:
            return _geojson(2, '2. Sand')
        return json.dumps({'features': []}).encode()

    monkeypatch.setattr(seabed, 'http_get', flaky)
    rdb = seabed.fetch_bottom_transect((0.0, 0.0), (1.0, 0.0), n_points=3)
    assert (rdb.halfspace_sound_speed == rdb.halfspace_sound_speed[0]).all()   # held across gaps


def test_fetch_bottom_transect_all_uncovered_raises(monkeypatch):
    monkeypatch.setattr(seabed, 'http_get',
                        lambda url, **kw: json.dumps({'features': []}).encode())
    with pytest.raises(DataFetchError, match='along the transect'):
        seabed.fetch_bottom_transect((0.0, -140.0), (1.0, -140.0), n_points=3)


@pytest.mark.requires_network
def test_live_emodnet_north_sea():
    try:
        bp = seabed.fetch_bottom((56.0, 3.0))      # North Sea: sandy
    except DataFetchError as exc:
        pytest.skip(f"EMODnet unreachable: {exc.message}")
    assert isinstance(bp, BoundaryProperties)
    assert bp.sound_speed > 1500.0


def test_undefined_folk_class_is_refused_as_a_data_gap():
    # folk_5cl = 6 is real: 70 polygons carry it in the cached 1:1M layer, so
    # its refusal must read as a coverage gap, not as a schema change.
    with pytest.raises(DataFetchError, match='does not define'):
        seabed._bottom_from_folk5(seabed._FOLK5_UNCLASSIFIED, 54.6, 5.3,
                                  roughness=0.0)


def test_out_of_legend_folk_class_is_refused_as_unrecognised():
    with pytest.raises(DataFetchError, match='unrecognised Folk-5'):
        seabed._bottom_from_folk5(99, 54.6, 5.3, roughness=0.0)


def test_folk_class_5_keeps_the_rock_shear_pair():
    # Hard substrata route to the limestone preset; both shear fields travel.
    bottom = seabed._bottom_from_folk5(5, 54.6, 5.3, roughness=0.0)
    assert bottom.shear_speed == pytest.approx(1500.0)
    assert bottom.shear_attenuation == pytest.approx(0.2)
