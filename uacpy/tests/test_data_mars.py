"""Tests for the AusSeabed MARS sediment fetch (uacpy.data.mars).

Stubs the WFS with canned GeoJSON so the grain-size / percentage / Folk-class
conversion chain, the nearest-usable-sample pick, the search-radius ladder and
the ``fetch_environment`` wiring run offline; one ``requires_network`` test
hits the live service.
"""

import json
import warnings

import numpy as np
import pytest

import uacpy.data as data
from uacpy.core.exceptions import DataFetchError
from uacpy.data import mars
from uacpy.data.environment import _AUTO_BOTTOM_ORDER
from uacpy.tests._cache_builders import _skip_or_fail


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
    # 'S' is Folk's sand class, not pure sand: its centroid carries 5 % mud, so
    # it converts a little finer than the sand end member's own 1.5 ϕ. The
    # 2e-4 slack is the class's own band, which reaches 0.01 % gravel.
    _install(monkeypatch, _collection(_feature(-34.0, 151.31, folk='S')))
    s = mars.fetch_mars_sediment(_P)
    assert s['phi'] == pytest.approx(0.95 * 1.5 + 0.05 * 7.5, abs=1e-3)
    assert s['via'] == 'folk_class'


def test_every_folk_class_phi_is_its_centroid_through_the_percentage_route():
    """The rule the table is built from, not the fifteen numbers it produces.

    Each class's ϕ is that class's own centroid on Folk's ternary diagram put
    through the *percentage* conversion, so a sample labelled ``'sG'`` and a
    sample whose measured percentages sit at the centroid of ``'sG'`` come back
    with the same ϕ. Asserting the numbers would only pin the arithmetic that
    produced them; this pins that the two routes cannot drift apart."""
    for (g_lo, g_hi), cells in mars._FOLK_CLASS_LIMITS.items():
        for x_lo, x_hi, code in cells:
            g, s, m = mars._folk_class_centroid(g_lo, g_hi, x_lo, x_hi)
            through_percentages = mars._phi_from_properties({
                'GRAVEL_PERCENT': 100.0 * g, 'SAND_PERCENT': 100.0 * s,
                'MUD_PERCENT': 100.0 * m})
            assert through_percentages['via'] == 'percentages'
            assert through_percentages['phi'] == pytest.approx(
                mars._FOLK_TO_PHI[code], abs=1e-9), code


def test_the_class_centroids_are_area_centroids_of_the_ternary_diagram():
    """The centroid rule itself, by a method that assumes none of it.

    A uniform grid over the triangle carries the diagram's own area measure, so
    averaging the compositions that fall in a class reproduces its ϕ without
    the closed form, and without the ``(1 - g) dg dx`` weight that form
    assumes. A plain midpoint of the class's boundaries would miss by tenths of
    a ϕ; the four zero-gravel classes are the diagram's base edge and hold no
    grid points, so they are left to the test above."""
    step = 0.002
    axis = np.arange(step / 2.0, 1.0, step)
    gg, ss = np.meshgrid(axis, axis, indexing='ij')
    inside = (gg + ss) < 1.0
    g, s = gg[inside], ss[inside]
    m = 1.0 - g - s
    x = s / (s + m)
    checked = 0
    for (g_lo, g_hi), cells in mars._FOLK_CLASS_LIMITS.items():
        for x_lo, x_hi, code in cells:
            sel = (g >= g_lo) & (g < g_hi) & (x >= x_lo) & (x < x_hi)
            if sel.sum() < 200:
                continue
            phi = mars._phi_of_mixture(g[sel].mean(), s[sel].mean(),
                                       m[sel].mean())
            assert phi == pytest.approx(mars._FOLK_TO_PHI[code], abs=0.01), code
            checked += 1
    assert checked == 11, f"the grid resolved {checked} classes, not 11"


def test_the_class_limits_are_folks_published_thresholds():
    """The tiling test below would pass just as well with 0.25 in place of
    0.30: it checks that the bands partition the diagram, not that they are
    *Folk's* bands. These are the published numbers, and every derived ϕ moves
    with them — silently and consistently, which is what makes them worth
    pinning here.

    Folk's gravel lines are 80 / 30 / 5 / 0.01 weight percent and his sand-mud
    lines are the ratios 1:9, 1:1 and 9:1, so x = s/(s+m) cuts at 0.1, 0.5,
    0.9. Which ratio lines apply depends on the band: the 1:9 cut exists only
    below 5 % gravel, leaving three classes in each of the two upper bands and
    four in each of the two lower ones. Read off the figure in USGS Open-File
    Report 2006-1195 (Nomenclature > Folk) and USGS Scientific Investigations
    Report 2019-5073 (fig. 3), whose text gives the thresholds in prose and
    cites Folk (1954, fig. 1a, table 1; 1980, p. 25-28, table 1)."""
    assert mars._FOLK_CLASS_LIMITS == {
        (0.80, 1.00): ((0.0, 1.0, 'G'),),
        (0.30, 0.80): ((0.0, 0.5, 'mG'), (0.5, 0.9, 'msG'), (0.9, 1.0, 'sG')),
        (0.05, 0.30): ((0.0, 0.5, 'gM'), (0.5, 0.9, 'gmS'), (0.9, 1.0, 'gS')),
        (0.0001, 0.05): ((0.0, 0.1, '(g)M'), (0.1, 0.5, '(g)sM'),
                         (0.5, 0.9, '(g)mS'), (0.9, 1.0, '(g)S')),
        (0.0, 0.0001): ((0.0, 0.1, 'M'), (0.1, 0.5, 'sM'),
                        (0.5, 0.9, 'mS'), (0.9, 1.0, 'S')),
    }


def test_folks_bands_and_ratio_cells_tile_the_ternary():
    """A gap would drop compositions the percentage route still converts; an
    overlap would make a class depend on dict order. The gravel bands partition
    the whole gravel axis and each band's ratio cells partition the sand:mud
    axis, which is what makes every class's centroid well defined."""
    bands = sorted(mars._FOLK_CLASS_LIMITS)
    assert bands[0][0] == 0.0 and bands[-1][1] == 1.0
    for (_, hi), (lo, _) in zip(bands, bands[1:]):
        assert hi == lo, f"gravel bands meet badly at {hi} / {lo}"
    for cells in mars._FOLK_CLASS_LIMITS.values():
        edges = sorted((lo, hi) for lo, hi, _ in cells)
        assert edges[0][0] == 0.0 and edges[-1][1] == 1.0
        for (_, hi), (lo, _) in zip(edges, edges[1:]):
            assert hi == lo, f"ratio cells meet badly at {hi} / {lo}"
    codes = [c for cells in mars._FOLK_CLASS_LIMITS.values()
             for _, _, c in cells]
    assert len(codes) == len(set(codes)) == 15


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


@pytest.mark.parametrize('folk, model', [('G', 'hamilton'),    # -1.13 ϕ
                                         ('sG', 'hamilton'),   # -0.11 ϕ
                                         ('G', 'apl-uw')])     # -1.13 < -1
def test_a_sample_coarser_than_the_relations_says_so(monkeypatch, folk, model):
    """MARS reaches seabed coarser than either grain-size relation is fitted
    over, and the conversion answers those with its fit at the nearer end —
    under the default 'hamilton' that is the 0.92 ϕ coarse-sand row, which
    nothing downstream can tell from a real coarse sand. The sample's own ϕ
    still travels on the bottom; only the geoacoustics are the fit's."""
    _install(monkeypatch, _collection(_feature(-34.0, 151.31, folk=folk)))
    with pytest.warns(UserWarning, match='fitted over'):
        bp = mars.fetch_bottom_mars(_P, model=model)
    assert bp.grain_size_phi == pytest.approx(mars._FOLK_TO_PHI[folk])
    edge = data.bottom_from_grain_size(
        mars.GRAIN_SIZE_MODEL_RANGES[model][0], model=model)
    assert bp.sound_speed == pytest.approx(edge.sound_speed)


@pytest.mark.parametrize('folk, model', [('sG', 'apl-uw'),     # -0.11 ϕ
                                         ('S', 'hamilton')])   # 1.80 ϕ
def test_a_sample_inside_the_relations_is_converted_in_silence(
        monkeypatch, folk, model):
    """The other side of each model's own range, and 'sG' is the pair that
    matters: at -0.11 ϕ it sits below 'hamilton''s 0 and inside 'apl-uw''s
    -1, so the same sample is announced by one model and converted in silence
    by the other."""
    _install(monkeypatch, _collection(_feature(-34.0, 151.31, folk=folk)))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        bp = mars.fetch_bottom_mars(_P, model=model)
    assert bp.grain_size_phi == pytest.approx(mars._FOLK_TO_PHI[folk])


@pytest.mark.parametrize('gravel, sand, warns', [(50.0, 50.0, True),    # -0.25 ϕ
                                                 (40.0, 60.0, False)])  # +0.10 ϕ
def test_the_announcement_turns_over_at_the_models_own_edge(
        monkeypatch, gravel, sand, warns):
    """Both sides of the threshold itself, which no Folk class lands on: the
    percentage route makes ϕ continuous, so two mixtures either side of
    'hamilton''s 0 ϕ separate the samples it can convert from the ones it
    answers with its end row."""
    _install(monkeypatch, _collection(
        _feature(-34.0, 151.31, gravel=gravel, sand=sand, mud=0.0)))
    if warns:
        with pytest.warns(UserWarning, match='fitted over'):
            mars.fetch_bottom_mars(_P)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            mars.fetch_bottom_mars(_P)


def test_bottom_from_mars_stamps_sample_provenance(monkeypatch):
    """Regression: a MARS hit can be up to max_distance_km from the request,
    so the bottom must carry a ``DataProvenance`` with the sample's own
    coordinates — the stamp the local grain-size DB already carries —
    rather than a bare ``BoundaryProperties``."""
    _install(monkeypatch, _collection(_feature(-34.4, 151.31, grain_um=500.0)))
    bp = mars.fetch_bottom_mars(_P)
    assert [p.source.id for p in bp.data_sources] == ['mars']
    prov = bp.data_sources[0]
    assert prov.data_point == (-34.4, 151.31)
    assert prov.requested_point == _P
    assert prov.offset_km == pytest.approx(44.5, abs=0.5)


def test_bottom_transect(monkeypatch):
    _install(monkeypatch, _collection(_feature(-34.0, 151.31, grain_um=500.0)))
    bottom = mars.fetch_bottom_mars_transect(
        (-34.0, 151.3), (-34.0, 151.8), n_points=3)
    assert np.allclose(bottom.halfspace_sound_speed,
                       bottom.halfspace_sound_speed[0])
    # Each rebuilt column carries the sample's provenance stamp.
    for col in bottom.columns:
        assert [p.source.id for p in col.data_sources] == ['mars']


# ── registry wiring ─────────────────────────────────────────────────────────

def test_mars_in_auto_chain_after_diesing():
    """MARS is live-only, so the installed Diesing raster is consulted first."""
    order = list(_AUTO_BOTTOM_ORDER)
    assert 'mars' in order
    assert order.index('diesing') < order.index('mars') < order.index('pelagic')


# ── coverage guard ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('point', [
    (43.2, 7.5),            # Ligurian Sea
    (36.0, -70.0),          # North Atlantic
    (26.0, -90.0),          # Gulf of Mexico shelf
    (60.0, 150.0),          # Sea of Okhotsk (right longitude, wrong hemisphere)
])
def test_out_of_coverage_never_hits_the_network(monkeypatch, point):
    def boom(url, **kw):
        raise AssertionError(f"MARS issued a request for {point}: {url}")

    monkeypatch.setattr(mars, 'http_get', boom)
    with pytest.raises(DataFetchError, match='does not cover'):
        mars.fetch_mars_sediment(point)


def test_auto_chain_reaches_no_mars_request_outside_australia(monkeypatch,
                                                              tmp_path):
    """The 'auto' bottom chain must not call AusSeabed for a Ligurian point."""
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    import uacpy.data.pelagic as pelagic_mod
    import uacpy.data.seabed as seabed_mod

    def no_emodnet(point, **kw):
        raise DataFetchError("European seas only")

    def boom(url, **kw):
        raise AssertionError(f"MARS issued a request: {url}")

    monkeypatch.setattr(seabed_mod, 'fetch_bottom', no_emodnet)
    monkeypatch.setattr(pelagic_mod, '_water_depth', lambda *a, **k: 5000.0)
    monkeypatch.setattr(mars, 'http_get', boom)
    env = data.fetch_environment((43.2, 7.5), bathymetry=2000.0, ssp=1500.0,
                                 bottom_sources='auto')
    assert env.data_sources[-1].source.id == 'pelagic'


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
        _skip_or_fail(exc, 'AusSeabed WFS')
    assert -5.0 < s['phi'] < 13.0


# ── OFFLINE stubs: nearest-sample correctness ───────────────────────────────

def test_radius_ladder_corner_sample_keeps_expanding(monkeypatch):
    # Rung 1 (10 km) bbox corner holds a ~13.8 km sample — beyond the rung
    # radius — while a closer ~11 km sample sits due east, outside that bbox.
    # The ladder must keep expanding instead of settling for the corner hit.
    corner = _feature(-34.088, 151.405, grain_um=1000.0)     # ϕ = 0, 13.8 km
    closer = _feature(-34.0, 151.4196, grain_um=250.0)       # ϕ = 2, 11.0 km
    calls = []

    def responder(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            return _collection(corner)
        return _collection(corner, closer)

    _install(monkeypatch, responder)
    s = mars.fetch_mars_sediment(_P)
    assert len(calls) > 1
    assert s['phi'] == pytest.approx(2.0)
    assert s['distance_km'] == pytest.approx(11.0, abs=0.2)


def test_corner_sample_confirmed_by_wider_rung(monkeypatch):
    # A rung-1 corner hit beyond the rung radius is re-checked at the wider
    # rung; with nothing closer there, it is still the returned sample.
    calls = []

    def responder(url, **kw):
        calls.append(url)
        return _collection(_feature(-34.088, 151.405, grain_um=1000.0))

    _install(monkeypatch, responder)
    s = mars.fetch_mars_sediment(_P)
    assert len(calls) == 2                       # 13.8 km ≤ 30 km ends rung 2
    assert s['phi'] == pytest.approx(0.0)
    assert s['distance_km'] == pytest.approx(13.76, abs=0.1)


def test_capped_response_warns_not_nearest(monkeypatch):
    # numberMatched above the returned feature count means the server capped
    # the page — the true nearest sample may be missing from the response.
    body = json.dumps({
        'type': 'FeatureCollection',
        'features': [_feature(-34.0, 151.31, grain_um=500.0)],
        'numberMatched': 5000,
    }).encode()
    _install(monkeypatch, body)
    with pytest.warns(UserWarning, match='nearest'):
        s = mars.fetch_mars_sediment(_P)
    assert s['phi'] == pytest.approx(1.0)


def _mars_payload(properties):
    return json.dumps({'type': 'FeatureCollection', 'numberMatched': 1,
                       'features': [{
                           'type': 'Feature',
                           'geometry': {'type': 'Point',
                                        'coordinates': [151.4, -33.9]},
                           'properties': properties}]})


def _patch_mars_http(monkeypatch, properties):
    _install(monkeypatch, _mars_payload(properties))
    return mars


@pytest.mark.parametrize('properties', [
    {'MEAN_GRAIN_SIZE': 'N/A'},
    {'SAND_PERCENT': 'trace'},
    {'MEAN_GRAIN_SIZE': [125.0]},
], ids=['string-grain-size', 'string-percent', 'list-grain-size'])
def test_a_malformed_mars_property_raises_the_typed_fetch_error(
        monkeypatch, properties):
    mars = _patch_mars_http(monkeypatch, properties)
    with pytest.raises(DataFetchError, match='non-numeric'):
        mars.fetch_mars_sediment((-33.9, 151.5))


def test_the_bottom_chain_falls_through_a_malformed_mars_sample(monkeypatch):
    from uacpy.data import environment as env_mod
    _patch_mars_http(monkeypatch, {'MEAN_GRAIN_SIZE': 'N/A'})
    _bottom, source = env_mod._fetch_bottom(
        ('mars', 'pelagic'), (-33.9, 151.5), transect=False,
        max_distance_km=100.0, depth=1000.0, timeout=5.0, verbose=False)
    assert source == 'pelagic'
