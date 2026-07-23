"""Tests for the AusSeabed MARS sediment fetch (uacpy.data.mars).

Stubs the WFS with canned GeoJSON so the grain-size / percentage / Folk-class
conversion chain, the nearest-usable-sample pick, the search-radius ladder and
the ``fetch_environment`` wiring run offline; one ``requires_network`` test
hits the live service.
"""

import json

import numpy as np
import pytest

import uacpy.data as data
from uacpy.core.exceptions import DataFetchError
from uacpy.data import mars
from uacpy.data.environment import _AUTO_BOTTOM_ORDER


def _feature(lat, lon, *, grain_um=None, mud=None, sand=None, gravel=None,
             folk=None):
    return {
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [lon, lat, -50]},
        'properties': {
            'MEAN_GRAIN_SIZE': grain_um,
            'MUD_PERCENT': mud, 'SAND_PERCENT': sand, 'GRAVEL_PERCENT': gravel,
            'FOLK_CLASS': folk,
        },
    }


def _collection(*features):
    return json.dumps({'type': 'FeatureCollection',
                       'features': list(features),
                       'numberMatched': len(features)}).encode()


def _install(monkeypatch, body):
    """Point the module's http_get at a canned (or callable) response."""
    fn = body if callable(body) else (lambda url, **kw: body)
    monkeypatch.setattr(mars, 'http_get', fn)


_P = (-34.0, 151.3)                                    # off Sydney


# ── conversion chain ────────────────────────────────────────────────────────

def test_grain_size_um_to_phi(monkeypatch):
    # 500 µm = 0.5 mm → ϕ = -log2(0.5) = 1.0; grain size beats the other fields.
    _install(monkeypatch, _collection(
        _feature(-34.0, 151.31, grain_um=500.0, mud=100.0, folk='M')))
    s = mars.fetch_mars_sediment(_P)
    assert s['phi'] == pytest.approx(1.0)
    assert s['via'] == 'grain_size'


def test_percentages_fallback(monkeypatch):
    # No grain size → gravel/sand/mud weighted ϕ: 50 % sand + 50 % mud → 4.5.
    _install(monkeypatch, _collection(
        _feature(-34.0, 151.31, mud=50.0, sand=50.0, gravel=0.0)))
    s = mars.fetch_mars_sediment(_P)
    assert s['phi'] == pytest.approx(0.5 * 1.5 + 0.5 * 7.5)
    assert s['via'] == 'percentages'


def test_folk_class_fallback(monkeypatch):
    _install(monkeypatch, _collection(_feature(-34.0, 151.31, folk='S')))
    s = mars.fetch_mars_sediment(_P)
    assert s['phi'] == pytest.approx(1.5)
    assert s['via'] == 'folk_class'


def test_unknown_folk_class_is_unusable(monkeypatch):
    _install(monkeypatch, _collection(_feature(-34.0, 151.31, folk='??')))
    with pytest.raises(DataFetchError, match='no usable sediment sample'):
        mars.fetch_mars_sediment(_P)


# ── nearest pick + guards ───────────────────────────────────────────────────

def test_picks_nearest_usable(monkeypatch):
    # The nearest feature has no usable fields; the next-nearest wins.
    _install(monkeypatch, _collection(
        _feature(-34.001, 151.301),                      # closest, unusable
        _feature(-34.05, 151.35, grain_um=1000.0),       # usable → ϕ = 0
        _feature(-34.5, 151.8, grain_um=62.5),           # farther
    ))
    s = mars.fetch_mars_sediment(_P)
    assert s['phi'] == pytest.approx(0.0)
    assert s['distance_km'] < 10.0


def test_no_coverage_raises(monkeypatch):
    _install(monkeypatch, _collection())
    with pytest.raises(DataFetchError, match='no usable sediment sample'):
        mars.fetch_mars_sediment(_P)


def test_max_distance_guard(monkeypatch):
    # Only sample is ~55 km away; a 20 km guard rejects it.
    _install(monkeypatch, _collection(
        _feature(-34.5, 151.3, grain_um=500.0)))
    with pytest.raises(DataFetchError, match='max_distance_km'):
        mars.fetch_mars_sediment(_P, max_distance_km=20.0)


def test_radius_ladder_expands(monkeypatch):
    # First (small-radius) query returns nothing; the wider retry finds the
    # sample — the fetch must not give up after the first empty bbox.
    calls = []

    def responder(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            return _collection()
        return _collection(_feature(-34.3, 151.3, grain_um=250.0))

    _install(monkeypatch, responder)
    s = mars.fetch_mars_sediment(_P)
    assert len(calls) > 1
    assert s['phi'] == pytest.approx(2.0)


# ── bottom builders ─────────────────────────────────────────────────────────

def test_bottom_from_mars(monkeypatch):
    _install(monkeypatch, _collection(_feature(-34.0, 151.31, grain_um=500.0)))
    bp = mars.fetch_bottom_mars(_P)
    assert bp.acoustic_type == 'half-space'
    assert bp.grain_size_phi == pytest.approx(1.0)
    assert bp.sound_speed > 1510.0                       # coarse sand: faster


def test_bottom_transect(monkeypatch):
    _install(monkeypatch, _collection(_feature(-34.0, 151.31, grain_um=500.0)))
    bottom = mars.fetch_bottom_mars_transect(
        (-34.0, 151.3), (-34.0, 151.8), n_points=3)
    assert np.allclose(bottom.halfspace_sound_speed,
                       bottom.halfspace_sound_speed[0])


# ── registry wiring ─────────────────────────────────────────────────────────

def test_mars_in_auto_chain_before_diesing():
    order = list(_AUTO_BOTTOM_ORDER)
    assert 'mars' in order
    assert order.index('emodnet') < order.index('mars') < order.index('diesing')


def test_fetch_environment_bottom_sources_mars(monkeypatch, tmp_path):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    _install(monkeypatch, _collection(_feature(-34.0, 151.31, grain_um=500.0)))
    env = data.fetch_environment(_P, bathymetry=1000.0, ssp=1500.0,
                                 bottom_sources='mars')
    assert env.bottom.halfspace_sound_speed[0] > 1510.0
    assert 'mars' in [s.source.id for s in env.data_sources]


# ── live ────────────────────────────────────────────────────────────────────

@pytest.mark.requires_network
def test_live_mars_point():
    try:
        s = mars.fetch_mars_sediment((-34.0, 151.5))     # off Sydney
    except DataFetchError as exc:
        pytest.skip(f"AusSeabed WFS unreachable: {exc.message}")
    assert -5.0 < s['phi'] < 13.0
