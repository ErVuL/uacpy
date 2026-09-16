"""Tests for the sediment model: grain size in, geoacoustics out.

uacpy.core.sediment holds the conversion itself and uacpy.data.sediment
the lookups and transect assembly built on it; the former is re-exported
through the latter, so both halves are one subject and live here together.

The conversion uses the Hamilton & Bachman (1982) continental-terrace relations
(density / sound-speed ratios, water-referenced) plus the Hamilton (1972) k_p
attenuation law of Geophysics 37, Fig. 3.

Each returned quantity has a validity range in phi, and it is its *source's*
range rather than the model's: the subject of the last class. 'hamilton' is two
sources, so one call can interpolate the attenuation while holding the velocity
and density at a table end row, and the report has to name which — otherwise the
caller cannot tell a substituted answer from an evaluated one.
"""

import importlib
import inspect
import pkgutil
import warnings

import numpy as np
import pytest

import uacpy

from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core.materials import MATERIALS
from uacpy.core.sediment import (
    DEFAULT_GRAIN_SIZE_MODEL, GRAIN_SIZE_MODEL_RANGES, GRAIN_SIZE_MODELS,
    GRAIN_SIZE_SOURCE_RANGES, _MODEL_WATER_REFERENCE, _apl_density_ratio,
    _apl_velocity_ratio, _hamilton_kp, grain_size_to_geoacoustics,
)
from uacpy.data import sediment
from uacpy.sonar.bottom_scattering import (_grain_size_alpha_over_f,
                                           _grain_size_density_ratio,
                                           _grain_size_speed_ratio)


def test_the_regression_reproduces_the_class_means_it_was_fitted_to():
    """Hamilton & Bachman published a table of class means and, in their
    Appendix, regressions fitted to the same continental-terrace dataset. uacpy
    evaluates the regressions, because the paper says to when a mean grain size
    is what you hold (p. 1892); the table is kept as data, and it is also the
    check on the fit. Inside the equations' declared 1-9 ϕ the two describe one
    dataset, well within the published σ of 29 m/s and 0.11 g/cm³ — and the one
    row that misses badly is the one row below 1 ϕ, which is Hamilton's own
    limit showing rather than a defect."""
    from uacpy.core.sediment import _HB_TABLE
    dc, drho, coarse_sand = [], [], None
    for phi, rho_row, ratio_row in _HB_TABLE:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got = sediment.grain_size_to_geoacoustics(phi)
        gap = (got['sound_speed'] - ratio_row * 1510.0, got['density'] - rho_row)
        if phi >= 1.0:
            dc.append(gap[0])
            drho.append(gap[1])
        else:
            coarse_sand = gap
    assert np.sqrt(np.mean(np.square(dc))) < 29.0       # published sigma
    assert max(abs(x) for x in dc) < 29.0
    assert max(abs(x) for x in drho) < 0.11             # published sigma
    assert abs(coarse_sand[0]) > 3 * np.sqrt(np.mean(np.square(dc)))


def test_the_class_mean_table_is_kept_as_data():
    """Not deleted with the interpolation it used to feed: these are the class
    means the regressions were fitted to, carrying the clayey-silt *median*
    that Table II's footnote recommends for that one class, and the rows
    ``uacpy.data.graw_local`` inverts for ρ → ϕ."""
    from uacpy.core.sediment import _HB_TABLE
    assert _HB_TABLE[0] == (0.92, 2.034, 1.201)         # coarse sand
    assert _HB_TABLE[-2] == (7.13, 1.484, 1.006)        # clayey silt, median
    assert _HB_TABLE[-1] == (8.80, 1.480, 0.990)        # silty clay


def test_velocity_ratio_dips_below_water_for_mud():
    # Fine muds are slower than seawater (velocity ratio < 1).
    fine = sediment.grain_size_to_geoacoustics(8.5, water_sound_speed=1500.0)
    assert fine['sound_speed'] < 1500.0


def test_monotonic_speed_and_density():
    coarse = sediment.grain_size_to_geoacoustics(1.0)
    fine = sediment.grain_size_to_geoacoustics(8.0)
    assert coarse['sound_speed'] > fine['sound_speed']
    assert coarse['density'] > fine['density']


def test_attenuation_peaks_at_four_and_a_half_phi():
    # Hamilton's k_p attenuation peaks at the 4.5 ϕ branch join — coarse silt
    # in TR 9407 Table 2 — and the dB/λ speed factor does not move it.
    phis = np.linspace(0.5, 8.5, 33)
    alpha = [sediment.grain_size_to_geoacoustics(p)['attenuation'] for p in phis]
    peak_phi = phis[int(np.argmax(alpha))]
    assert peak_phi == pytest.approx(4.5, abs=0.13)   # half the 0.25 ϕ step


def test_water_referencing_scales_speed():
    warm = sediment.grain_size_to_geoacoustics(2.5, water_sound_speed=1540.0)
    cold = sediment.grain_size_to_geoacoustics(2.5, water_sound_speed=1480.0)
    assert warm['sound_speed'] > cold['sound_speed']


def test_a_phi_past_the_regressions_domain_is_held_at_it_and_says_so():
    # Past 9 ϕ the (T) quadratics are not evaluated — Hamilton declares them to
    # 9 ϕ and there is no environment-free answer beyond it — so ϕ is held at
    # the edge, and the call names the quantities that came from there.
    with pytest.warns(UserWarning, match='sound_speed and density'):
        g = sediment.grain_size_to_geoacoustics(12.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        edge = sediment.grain_size_to_geoacoustics(9.0)
    assert g['sound_speed'] == pytest.approx(edge['sound_speed'])
    assert g['density'] == pytest.approx(edge['density'])


def test_out_of_range_phi_warns_when_the_clamp_moves_the_answer():
    # APL-UW's polynomials do extrapolate, so clamping ϕ = 12 to 9 substitutes
    # a different sediment and says so.
    with pytest.warns(UserWarning, match='clamped'):
        g = sediment.grain_size_to_geoacoustics(12.0, model='apl-uw')
    assert g['sound_speed'] == pytest.approx(
        sediment.grain_size_to_geoacoustics(9.0, model='apl-uw')['sound_speed'])


def test_unknown_model_raises():
    with pytest.raises(ConfigurationError, match='model'):
        sediment.grain_size_to_geoacoustics(3.0, model='nonsense')


def test_apl_uw_model():
    # APL-UW TR 9407 high-frequency variant: valid output, attenuation peaking
    # at the same 4.5 ϕ join, and (per IV-8) lower density/speed than Hamilton
    # at intermediate Mz.
    a = sediment.grain_size_to_geoacoustics(2.0, model='apl-uw')
    h = sediment.grain_size_to_geoacoustics(2.0, model='hamilton')
    assert 1400 < a['sound_speed'] < 1800
    assert a['density'] < h['density']            # APL-UW runs lower at mid-Mz
    phis = np.linspace(-1, 9, 41)
    alpha = [sediment.grain_size_to_geoacoustics(p, model='apl-uw')['attenuation']
             for p in phis]
    assert phis[int(np.argmax(alpha))] == pytest.approx(4.5, abs=0.13)
    # APL-UW covers coarse (gravel ϕ≈−1) without warning
    coarse = sediment.grain_size_to_geoacoustics(-1.0, model='apl-uw')
    assert coarse['sound_speed'] > a['sound_speed']


def test_bottom_from_grain_size():
    bp = sediment.bottom_from_grain_size(3.5, roughness=0.2)
    assert isinstance(bp, BoundaryProperties)
    assert bp.acoustic_type == 'half-space'   # universal: works in every model
    assert bp.grain_size_phi == 3.5           # retained as informational metadata
    assert bp.roughness == 0.2
    assert 1650.0 < bp.sound_speed < 1720.0    # very fine sand ≈ 1691 m/s


def test_bottom_from_class():
    bp = sediment.bottom_from_class('sand')
    assert bp.acoustic_type == 'half-space'
    assert bp.sound_speed == pytest.approx(MATERIALS['sand']['sound_speed'])
    assert bp.density == pytest.approx(MATERIALS['sand']['density'])


def test_bottom_from_class_unknown_raises():
    with pytest.raises(ConfigurationError, match='unknown sediment class'):
        sediment.bottom_from_class('mud')


@pytest.mark.parametrize('name', ['limestone', 'granite', 'sand', 'clay'])
def test_bottom_from_class_keeps_the_shear_pair(name):
    """Shear speed and shear attenuation are one property of the material and
    must travel together: a rock half-space with a shear speed but *lossless*
    shear (limestone's alpha_s is 0.2 dB/lambda, sand's 2.5) under-predicts
    bottom loss in Kraken / Scooter / OASES."""
    bp = sediment.bottom_from_class(name)
    assert bp.shear_speed == pytest.approx(MATERIALS[name]['shear_speed'])
    assert bp.shear_attenuation == pytest.approx(
        MATERIALS[name]['shear_attenuation'])
    assert bp.name == name


def test_bottom_from_class_fluid_drops_both_shear_fields():
    bp = sediment.bottom_from_class('limestone', elastic=False)
    assert bp.shear_speed == 0.0
    assert bp.shear_attenuation == 0.0
    assert bp.sound_speed == pytest.approx(MATERIALS['limestone']['sound_speed'])


def test_grain_size_none_water_uses_hamilton_reference():
    """``water_sound_speed=None`` means "use Hamilton's own reference water"
    (1510 m/s, 1.030 g/cm³), so it must agree exactly with passing those two
    values explicitly — ``None`` is a default, not a separate code path."""
    explicit = sediment.grain_size_to_geoacoustics(1.0, water_sound_speed=1510.0,
                                                   water_density=1.030)
    default = sediment.grain_size_to_geoacoustics(1.0)
    assert default['sound_speed'] == pytest.approx(explicit['sound_speed'])
    assert default['density'] == pytest.approx(explicit['density'])


def test_grain_size_scales_with_in_situ_water_speed():
    """Sediment cp is a velocity *ratio* to the overlying water, so a
    colder/warmer in-situ water speed must shift the bottom cp proportionally
    instead of always referencing 1510 m/s."""
    cold = sediment.grain_size_to_geoacoustics(1.0, water_sound_speed=1450.0)
    warm = sediment.grain_size_to_geoacoustics(1.0, water_sound_speed=1540.0)
    ref = sediment.grain_size_to_geoacoustics(1.0)  # 1510 m/s reference
    assert cold['sound_speed'] < ref['sound_speed'] < warm['sound_speed']
    # Ratio is preserved: cp scales linearly with the water speed.
    ratio = ref['sound_speed'] / 1510.0
    assert warm['sound_speed'] == pytest.approx(ratio * 1540.0, rel=1e-6)


def test_range_dependent_bottom_preserves_shear():
    """``range_dependent_bottom_along`` must carry shear so an elastic (rock)
    waypoint is not silently flattened to a fluid half-space — dropping it
    changes the physics at that column, not just its resolution."""
    elastic = BoundaryProperties(
        acoustic_type='half-space', sound_speed=2500.0, density=2.0,
        attenuation=0.1, shear_speed=1200.0, shear_attenuation=0.2)
    bottom = sediment.range_dependent_bottom_along(
        lambda la, lo: elastic, (0.0, 0.0), (0.0, 0.1), 4,
        source_label='test')
    # Every column's half-space must retain the shear speed.
    for col in bottom.columns:
        assert col.halfspace.shear_speed == pytest.approx(1200.0)
        assert col.halfspace.shear_attenuation == pytest.approx(0.2)


def test_range_dependent_bottom_preserves_roughness():
    """A point-fetcher's roughness must survive the transect rebuild."""
    bp = BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0, density=1.9,
        attenuation=0.5, roughness=0.3)
    bottom = sediment.range_dependent_bottom_along(
        lambda la, lo: bp, (0.0, 0.0), (0.0, 0.1), 4, source_label='test')
    for col in bottom.columns:
        assert col.halfspace.roughness == pytest.approx(0.3)


def test_range_dependent_bottom_preserves_provenance():
    """Each sampled column's ``data_sources`` must survive the transect
    rebuild, so the assembled ``Bottom`` reports the same provenance a
    single-point fetch does."""
    from uacpy.data.sources import SOURCES, DataProvenance

    def point_bottom(la, lo):
        prov = DataProvenance(source=SOURCES['grainsize'],
                              data_point=(la, lo), requested_point=(la, lo))
        return BoundaryProperties(
            acoustic_type='half-space', sound_speed=1700.0, density=1.9,
            attenuation=0.5, data_sources=(prov,))

    bottom = sediment.range_dependent_bottom_along(
        point_bottom, (0.0, 0.0), (0.0, 0.1), 3, source_label='test')
    for col in bottom.columns:
        assert [p.source.id for p in col.data_sources] == ['grainsize']
    assert [p.source.id for p in bottom.data_sources] == ['grainsize']


def _phi_bottom(phi, water_sound_speed):
    def point_bottom(lat, lon):
        return sediment.bottom_from_grain_size(
            phi(lat) if callable(phi) else phi,
            water_sound_speed=sediment.water_sound_speed_at(
                water_sound_speed, lat, lon))
    return point_bottom


@pytest.mark.parametrize('water_sound_speed', [
    None,
    1500.0,
    lambda la, lo: 1500.0,
    lambda la, lo: 1480.0 + 40.0 * (la - 40.0),     # varies along the transect
])
def test_auto_collapses_uniform_seabed_under_varying_water_speed(
        water_sound_speed):
    """'auto' collapses a uniform seabed to one column whatever the water does.

    The grain-size geoacoustics are a ratio against the overlying water, so a
    range-dependent ``water_sound_speed`` callable makes every probe point's
    sound speed distinct. The collapse keys on the sediment (ϕ), not on that.
    """
    bottom = sediment.range_dependent_bottom_along(
        _phi_bottom(5.0, water_sound_speed), (40.0, -30.0), (41.0, -30.0),
        'auto', source_label='test', max_points=200)
    assert len(bottom.columns) == 1


def test_an_explicit_count_above_max_points_is_capped_with_a_warning():
    with pytest.warns(UserWarning, match=r'n_points=6 exceeds max_points=3'):
        bottom = sediment.range_dependent_bottom_along(
            _phi_bottom(5.0, 1500.0), (40.0, -30.0), (41.0, -30.0), 6,
            source_label='test', max_points=3)
    assert len(bottom.columns) == 3


def test_auto_splits_on_a_real_sediment_change():
    """The collapse must not merge distinct sediments (ϕ 3 → ϕ 7): each run
    keeps the probe columns bracketing its edges plus the endpoints."""
    bottom = sediment.range_dependent_bottom_along(
        _phi_bottom(lambda la: 3.0 if la < 40.5 else 7.0,
                    lambda la, lo: 1480.0 + 40.0 * (la - 40.0)),
        (40.0, -30.0), (41.0, -30.0), 'auto', source_label='test',
        max_points=200)
    assert len(bottom.columns) == 4
    speeds = [c.halfspace.sound_speed for c in bottom.columns]
    assert speeds[0] > speeds[2] and speeds[1] > speeds[3]  # ϕ3 pair, ϕ7 pair


def test_auto_places_the_transition_at_the_observed_boundary():
    """Regression: the nearest-node ``Bottom`` must rebuild a sediment change
    within one probe step of where the probe observed it. The former midpoint
    collapse anchored a two-run transect at its endpoints only, so the
    reconstructed boundary landed at mid-transect — hundreds of km off a real
    NSIDC ice edge / Diesing lithology boundary near one end."""
    probe_n = 200
    boundary_lat = 40.2                   # a fifth of the way along, not midway
    bottom = sediment.range_dependent_bottom_along(
        _phi_bottom(lambda la: 3.0 if la < boundary_lat else 7.0, None),
        (40.0, -30.0), (41.0, -30.0), 'auto', source_label='test',
        max_points=probe_n)
    length_m = bottom.ranges[-1]
    step_m = length_m / (probe_n - 1)
    true_edge_m = length_m * 0.2          # boundary_lat on the constant-lon path
    rho = [c.halfspace.density for c in bottom.columns]   # ϕ3 denser than ϕ7
    rr = np.asarray(bottom.ranges, dtype=float)
    transitions = [(rr[i] + rr[i + 1]) / 2.0
                   for i in range(len(rho) - 1) if rho[i] != rho[i + 1]]
    assert len(transitions) == 1
    assert abs(transitions[0] - true_edge_m) <= step_m
    # The nearest-node read agrees one probe step either side of the boundary.
    assert (bottom.at(range=true_edge_m - step_m).halfspace.density
            == pytest.approx(rho[0]))
    assert (bottom.at(range=true_edge_m + step_m).halfspace.density
            == pytest.approx(rho[-1]))
    assert rho[0] > rho[-1]


def test_auto_keys_on_geoacoustics_without_a_grain_size():
    """Sources reporting no ϕ (absolute crustal properties) key on the tuple."""
    def crustal(lat, lon):
        return BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0 if lat < 40.5 else 2200.0,
            density=2.0, attenuation=0.2)

    bottom = sediment.range_dependent_bottom_along(
        crustal, (40.0, -30.0), (41.0, -30.0), 'auto', source_label='test',
        max_points=200)
    assert ([c.halfspace.sound_speed for c in bottom.columns]
            == [1600.0, 1600.0, 2200.0, 2200.0])


def test_deck41_rock_routes_to_limestone_material(monkeypatch):
    """A 'rock' lithology cell carries the class sentinel through the phi
    index and comes back as the limestone material preset (~3000 m/s), the
    same route the EMODnet substrate path uses — never the coarse-sand
    clamp (~1813 m/s) a phi of -5 used to produce."""
    from uacpy.data import sediment_db
    sentinel = sediment_db._phi_from_lithology('rock')
    assert sentinel is not None
    assert sentinel <= sediment_db._PHI_CLASS_SENTINEL_MAX
    monkeypatch.setattr(
        sediment_db, 'fetch_sediment_sample',
        lambda point, max_distance_km=None: {
            'phi': None, 'material': 'limestone', 'distance_km': 1.0,
            'latitude': 0.0, 'longitude': 0.0})
    b = sediment_db.fetch_bottom_local((0.0, 0.0))
    assert b.sound_speed >= 2500.0


def test_deck41_gravel_classes_take_the_gravel_preset(tmp_path, monkeypatch):
    """'gravel' and 'gravel and coarser' are coarser than either grain-size
    relation is fitted over (both stop at -1 ϕ = 2 mm), so a ϕ for them is
    evaluated at the fit's coarse end: the default 'hamilton' answers every ϕ
    below 0 with its 0.92 ϕ coarse-sand row, which is neither lithology and is
    also what 'sand' returns. Both terms carry a class sentinel instead and
    come back as the gravel material preset, JKPS Table 1.3's own gravel. It
    stays fluid: Table 1.3 gives that c_s as a depth relation rather than a
    half-space property, and attaching it would make every fluid engine
    collapse the shear with a warning on a path that reports none."""
    from uacpy.data import _cache, sediment_db
    for term in ('gravel', 'gravel and coarser'):
        sentinel = sediment_db._phi_from_lithology(term)
        assert sentinel is not None
        assert sentinel <= sediment_db._PHI_CLASS_SENTINEL_MAX
    root = tmp_path / 'cache'
    (root / 'sediment').mkdir(parents=True)
    (root / 'sediment' / 'deck41.csv').write_text(
        'latitude,longitude,lithology\n'
        '10.0,20.0,gravel\n11.0,21.0,gravel and coarser\n')
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    _cache.invalidate_grids()
    sediment_db._SAMPLES.clear()
    coarse_sand = sediment.grain_size_to_geoacoustics(0.92)['sound_speed']
    for point in ((10.0, 20.0), (11.0, 21.0)):
        assert sediment_db.fetch_sediment_sample(point)['material'] == 'gravel'
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            bottom = sediment_db.fetch_bottom_local(point)
        assert bottom.sound_speed == MATERIALS['gravel']['sound_speed']
        assert bottom.density == MATERIALS['gravel']['density']
        assert bottom.attenuation == MATERIALS['gravel']['attenuation']
        assert bottom.sound_speed != pytest.approx(coarse_sand)
        assert bottom.shear_speed == 0.0


def test_sediment_csv_row_with_an_unreadable_longitude_raises(tmp_path):
    """The three sample lists are appended together, so a malformed longitude
    can no longer leave them at different lengths — that desync escaped as an
    untyped 'operands could not be broadcast together' from the KD-tree build,
    in the very file the docstring tells users to hand-write."""
    from uacpy.core.exceptions import DataFetchError
    from uacpy.data import sediment_db
    path = tmp_path / 'deck41.csv'
    path.write_text('latitude,longitude,phi\n1.0,2.0,3.0\n4.0,EAST,6.0\n')
    with pytest.raises(DataFetchError, match='line 3'):
        sediment_db._read_csv(path, ('phi',), sediment_db._phi_from_float)


def test_sediment_csv_skips_blank_and_valueless_rows(tmp_path):
    from uacpy.data import sediment_db
    path = tmp_path / 'deck41.csv'
    path.write_text('latitude,longitude,phi\n1.0,2.0,3.0\n\n'
                    '4.0,5.0,notaphi\n6.0,7.0,8.0\n')
    lats, lons, phis = sediment_db._read_csv(path, ('phi',),
                                             sediment_db._phi_from_float)
    assert lats == [1.0, 6.0]
    assert lons == [2.0, 7.0]
    assert phis == [3.0, 8.0]


@pytest.fixture
def sediment_cache(tmp_path, monkeypatch):
    """Cache with one grain-size sample and a *nearer* lithology sample."""
    from uacpy.data import sediment_db
    root = tmp_path / 'data_cache'
    (root / 'sediment').mkdir(parents=True)
    (root / 'sediment' / 'grainsize.csv').write_text(
        'latitude,longitude,mean_phi\n50.0,0.0,5.5\n')
    # A lithology term that carries a ϕ, so these tests pin which index
    # answered rather than the class-sentinel route gravel takes.
    (root / 'sediment' / 'deck41.csv').write_text(
        'latitude,longitude,lithology\n50.001,0.0,clay\n60.0,0.0,clay\n')
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    sediment_db._SAMPLES.clear()
    yield root
    sediment_db._SAMPLES.clear()


def test_a_quantitative_sample_beats_a_nearer_lithology_class(sediment_cache):
    """At comparable range the measured mean ϕ wins, even though the lithology
    word sits exactly on the query point — one merged KD-tree decided this on
    distance alone and never returned the better sample."""
    from uacpy.data.sediment_db import fetch_sediment_sample
    sample = fetch_sediment_sample((50.001, 0.0))
    assert sample['phi'] == pytest.approx(5.5)
    assert sample['distance_km'] == pytest.approx(0.111, abs=1e-3)


def test_a_far_quantitative_sample_loses_to_a_near_lithology_class(tmp_path,
                                                                   monkeypatch):
    """The preference is distance-aware, not absolute.

    A measurement 138.7 km away describes a different sediment province from
    the query; a class 1.4 km away describes this one. Preferring the
    measurement whatever the separation is a larger error than the merged tree
    it replaced — being quantitative about the wrong place beats nothing.
    """
    from uacpy.data import sediment_db
    root = tmp_path / 'data_cache'
    (root / 'sediment').mkdir(parents=True)
    (root / 'sediment' / 'grainsize.csv').write_text(
        'latitude,longitude,mean_phi\n43.0,7.0,2.0\n')
    (root / 'sediment' / 'deck41.csv').write_text(
        'latitude,longitude,lithology\n44.0,8.0,Sand\n')
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    sediment_db._SAMPLES.clear()
    try:
        sample = sediment_db.fetch_sediment_sample((44.01, 8.01))
        assert sample['phi'] == pytest.approx(1.5)          # 'Sand'
        assert sample['distance_km'] == pytest.approx(1.37, abs=0.01)
    finally:
        sediment_db._SAMPLES.clear()


def test_the_grain_size_preference_turns_over_between_the_two_anchors():
    """The rule itself, away from any cache: co-located keeps the measurement,
    a hundredfold hop does not."""
    from uacpy.data.sediment_db import _prefers_grain_size

    def hit(distance_km):
        return (distance_km, 0.0, 0.0, 0.0)

    assert _prefers_grain_size(hit(0.111), hit(0.0))     # same patch of seabed
    assert not _prefers_grain_size(hit(138.7), hit(1.37))
    assert _prefers_grain_size(hit(20.0), None)          # no class to compare
    # Past the co-location slack the comparison is relative, not absolute.
    assert _prefers_grain_size(hit(100.0), hit(50.0))
    assert not _prefers_grain_size(hit(200.0), hit(50.0))


def test_lithology_answers_where_grain_size_is_out_of_reach(sediment_cache):
    from uacpy.data.sediment_db import fetch_sediment_sample
    # 'clay' maps to ϕ 9.0; the grain-size sample is >1000 km away here.
    sample = fetch_sediment_sample((59.9, 0.0), max_distance_km=50.0)
    assert sample['phi'] == pytest.approx(9.0)


def test_the_out_of_reach_message_quotes_the_nearest_of_the_two(sediment_cache):
    from uacpy.data.sediment_db import fetch_sediment_sample
    with pytest.raises(DataFetchError, match='Nearest sediment sample'):
        fetch_sediment_sample((0.0, 0.0), max_distance_km=10.0)


def test_range_dependent_bottom_carries_grain_size_per_column():
    """``from_halfspaces`` builds from the geoacoustic arrays alone, so ϕ has
    to be copied onto the rebuilt columns like the provenance beside it."""
    from uacpy.core.bottom import BoundaryProperties
    from uacpy.data.sediment import range_dependent_bottom_along
    bottom = range_dependent_bottom_along(
        lambda lat, lon: BoundaryProperties.from_grain_size(
            3.0 if lat < 0.5 else 7.0),
        (0.0, 0.0), (1.0, 0.0), n_points=5, source_label='test')
    assert [c.halfspace.grain_size_phi for c in bottom.columns] == [
        3.0, 3.0, 7.0, 7.0, 7.0]


class TestTransectSeabedGapsAreNearestAndAnnounced:
    """A waypoint the source does not cover takes the seabed of the nearest
    covered waypoint, and the call says so.

    Forward-filling gave a gap bracketed by coverage on both sides the
    *earlier* sample even when the later one was far closer — a
    direction-dependent answer to a question that has none — and it did so
    silently, while every other substitution in this layer warns (the WOA23
    dry-cell hop, the NSIDC unobserved-cell hop, the SSP seafloor
    extrapolation). Measured on a 1579 km NE Atlantic transect where EMODnet
    covers the first 324 km: the remaining 1255 km all took one polygon's
    class, giving rho*c 2.883e6 against 2.256e6 kg m^-2 s^-1 (+27.8 %) for the
    seabed a point fetch of the far end returns from the next source.
    """

    @staticmethod
    def _props(phi):
        from uacpy.data.sediment import bottom_from_grain_size
        return bottom_from_grain_size(phi)

    def test_a_gap_takes_the_nearer_side_not_the_earlier_one(self):
        from uacpy.data.sediment import _fill_gaps_from_nearest
        near, far = self._props(2.0), self._props(7.0)
        # Covered at r=0 and r=10 km; the hole at 9 km is 1 km from the later
        # sample and 9 km from the earlier one.
        vals = [near, None, far]
        ranges = np.array([0.0, 9000.0, 10000.0])
        out, report = _fill_gaps_from_nearest(vals, ranges)
        assert out[1] is far, "the gap took the earlier sample, not the nearer"
        assert report is not None and report[0] == 1
        assert report[1] == pytest.approx(1.0)      # km to the filling sample

    def test_a_leading_gap_is_filled_too(self):
        from uacpy.data.sediment import _fill_gaps_from_nearest
        covered = self._props(3.0)
        out, report = _fill_gaps_from_nearest(
            [None, None, covered], np.array([0.0, 1000.0, 2000.0]))
        assert out[0] is covered and out[1] is covered
        assert report[0] == 2

    def test_full_coverage_reports_nothing(self):
        from uacpy.data.sediment import _fill_gaps_from_nearest
        vals = [self._props(2.0), self._props(3.0)]
        out, report = _fill_gaps_from_nearest(vals, np.array([0.0, 1000.0]))
        assert out == vals and report is None

    def test_a_partly_covered_transect_warns(self):
        from uacpy.core.exceptions import DataFetchError
        from uacpy.data.sediment import range_dependent_bottom_along
        covered = self._props(4.0)

        def point_bottom(lat, lon):
            if lon > -1.0:                     # coverage only near the start
                return covered
            raise DataFetchError('no coverage here')

        with pytest.warns(UserWarning, match='filled from the nearest'):
            range_dependent_bottom_along(
                point_bottom, (45.0, 0.0), (45.0, -5.0), n_points=6,
                source_label='test-source')


class TestALithologySampleCitesDeck41:
    """The two local sediment indices are different datasets with different
    licences and DOIs, and ``fetch_sediment_sample`` answers from whichever is
    nearer. Stamping ``SOURCES['grainsize']`` unconditionally reported a DECK41
    lithology description under the NCEI grain-size database's name and DOI
    (10.7289/V5G44N6W) — a citation for a dataset the value never touched.
    """

    @staticmethod
    def _cache(tmp_path, monkeypatch):
        from uacpy.data import _cache, sediment_db
        root = tmp_path / 'cache'
        (root / 'sediment').mkdir(parents=True)
        (root / 'sediment' / 'grainsize.csv').write_text(
            'latitude,longitude,mean_phi\n30.0,-40.0,3.0\n')
        (root / 'sediment' / 'deck41.csv').write_text(
            'latitude,longitude,lithology\n-20.0,-140.0,clay\n'
            '-20.05,-140.05,rock\n')
        monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
        _cache.invalidate_grids()
        sediment_db._SAMPLES.clear()
        return sediment_db

    @pytest.mark.parametrize('point, expected', [
        ((-20.0, -140.0), 'deck41'),        # lithology text -> phi
        ((-20.05, -140.05), 'deck41'),      # 'rock' -> the limestone preset
        ((30.0, -40.0), 'grainsize'),       # a measured mean phi
    ])
    def test_the_sample_names_the_index_it_came_from(self, point, expected,
                                                     tmp_path, monkeypatch):
        dB = self._cache(tmp_path, monkeypatch)
        assert dB.fetch_sediment_sample(point)['dataset'] == expected

    @pytest.mark.parametrize('point, expected', [
        ((-20.0, -140.0), 'deck41'),
        ((30.0, -40.0), 'grainsize'),
    ])
    def test_the_bottom_provenance_cites_that_index(self, point, expected,
                                                    tmp_path, monkeypatch):
        dB = self._cache(tmp_path, monkeypatch)
        bottom = dB.fetch_bottom_local(point)
        assert bottom.data_sources[0].source.id == expected


class TestEachQuantityIsReportedAgainstItsOwnSource:
    """A validity range belongs to the source a quantity comes from, not to
    the model. ``'hamilton'`` is two sources: ``sound_speed`` and ``density``
    interpolate ``_HB_TABLE``'s nine rows (0.92-8.8 ϕ), ``attenuation`` is the
    ``k_p`` regression (0-9.5 ϕ). So the answer to "was anything substituted?"
    differs *between quantities of one call*, and the report names which."""

    @pytest.mark.parametrize('phi,unclamped_cp', [(9.5, 1468.1883),
                                                  (-1.5, 2052.8599)])
    def test_apl_uw_warns_when_the_clamp_moves_the_sound_speed(
            self, phi, unclamped_cp):
        with pytest.warns(UserWarning, match='clamped'):
            out = grain_size_to_geoacoustics(phi, model='apl-uw')
        clamped = grain_size_to_geoacoustics(
            float(np.clip(phi, -1.0, 9.0)), model='apl-uw')
        assert out['sound_speed'] == pytest.approx(clamped['sound_speed'])
        assert out['sound_speed'] != pytest.approx(unclamped_cp, abs=1e-3)

    def test_the_warning_reports_the_value_it_replaced(self):
        """Naming only the ϕ leaves the reader unable to judge the size of the
        substitution; ϕ = 9.5 moves the sound speed by 1.8 m/s."""
        with pytest.warns(UserWarning) as record:
            grain_size_to_geoacoustics(9.5, model='apl-uw')
        assert '1468.1883' in str(record[0].message)

    @pytest.mark.parametrize('phi', [-1.0, 9.0, 0.0, 8.999])
    def test_apl_uw_is_silent_inside_its_range(self, phi):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            grain_size_to_geoacoustics(phi, model='apl-uw')

    @pytest.mark.parametrize('phi', [9.0 + 1e-9, -1.0 - 1e-9])
    def test_apl_uw_warns_immediately_outside_its_range(self, phi):
        """The other side of the boundary the test above pins: there is no
        deadband left, so the first ϕ past the edge already warns."""
        with pytest.warns(UserWarning, match='clamped'):
            grain_size_to_geoacoustics(phi, model='apl-uw')

    @pytest.mark.parametrize('phi', [9.0001, 9.3, 9.799, 9.801, -1.0001,
                                     20.0, -20.0])
    def test_hamilton_holds_its_regressions_at_their_edge_and_says_which(
            self, phi):
        """Past -1 or 9 ϕ the (T) quadratics are held at the edge rather than
        run on, so the value returned is indistinguishable from one the source
        covers — which is why the call names the quantities that came from
        there. ϕ inside the union (9.0 to 9.5) still moves the attenuation,
        which is the case no single verdict describes."""
        lo, hi = GRAIN_SIZE_SOURCE_RANGES['hamilton']['sound_speed']
        with pytest.warns(UserWarning, match='sound_speed and density'):
            out = grain_size_to_geoacoustics(phi, model='hamilton')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            edge = grain_size_to_geoacoustics(
                float(np.clip(phi, lo, hi)), model='hamilton')
        assert out['sound_speed'] == pytest.approx(edge['sound_speed'])
        assert out['density'] == pytest.approx(edge['density'])

    @pytest.mark.parametrize('model, quantity', [
        ('hamilton', 'sound_speed'), ('hamilton', 'density'),
        ('hamilton', 'attenuation'), ('apl-uw', 'sound_speed'),
        ('apl-uw', 'density'), ('apl-uw', 'attenuation'),
    ])
    def test_a_phi_past_a_quantitys_own_source_is_announced_for_it(
            self, model, quantity):
        """The rule, read off ``GRAIN_SIZE_SOURCE_RANGES`` rather than written
        out: on either side of **each source's own** edge, the quantities it
        supplies are named the moment ϕ leaves it, and not before. Correcting a
        published domain moves the edge and this test with it."""
        lo, hi = GRAIN_SIZE_SOURCE_RANGES[model][quantity]
        for outside, inside in ((lo - 1e-9, lo), (hi + 1e-9, hi)):
            with pytest.warns(UserWarning, match=quantity):
                grain_size_to_geoacoustics(outside, model=model)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                grain_size_to_geoacoustics(inside, model=model)
            assert not [w for w in caught if quantity in str(w.message)], (
                f"{model} {quantity} announced at ϕ={inside}, inside its own "
                f"source range {(lo, hi)}")

    def test_the_model_range_is_the_union_of_its_sources(self):
        """``GRAIN_SIZE_MODEL_RANGES`` is what the clamp uses and is derived
        from the per-quantity ranges, so the two cannot drift apart."""
        for model, quantities in GRAIN_SIZE_SOURCE_RANGES.items():
            assert GRAIN_SIZE_MODEL_RANGES[model] == (
                min(lo for lo, _ in quantities.values()),
                max(hi for _, hi in quantities.values()))
        # 'apl-uw' is one equation set (TR 9407 Eqs. 2-10), so its three
        # quantities share one domain and nothing here can differ between them.
        assert len(set(GRAIN_SIZE_SOURCE_RANGES['apl-uw'].values())) == 1

    def test_each_environment_is_evaluated_over_its_own_published_range(self):
        """Hamilton fits three environments and declares a different domain for
        each — (T) 1 to 9 ϕ, (H) and (P) 7 to 10 — so the report has to follow
        the environment, not just the model. The abyssal fits are straight
        lines over deep-water fine sediment and describe no sand, so a coarser
        ϕ is held at 7 and announced rather than extrapolated."""
        from uacpy.core.sediment import GRAIN_SIZE_ENVIRONMENTS
        assert (GRAIN_SIZE_ENVIRONMENTS['continental-terrace']['published_range']
                == (1.0, 9.0))
        for abyssal in ('abyssal-hill', 'abyssal-plain'):
            fit = GRAIN_SIZE_ENVIRONMENTS[abyssal]
            assert fit['published_range'] == (7.0, 10.0)
            # nothing extends the abyssal families the way TR 9407 extends (T)
            assert fit['evaluated_range'] == fit['published_range']
            with pytest.warns(UserWarning, match='7 to 10 ϕ'):
                sand = grain_size_to_geoacoustics(2.0, environment=abyssal)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                edge = grain_size_to_geoacoustics(7.0, environment=abyssal)
            assert sand['sound_speed'] == pytest.approx(edge['sound_speed'])
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                grain_size_to_geoacoustics(9.5, environment=abyssal)

    def test_the_abyssal_fits_land_nearer_the_measured_abyssal_rows(self):
        """What the selector is for. Hamilton & Bachman's Table IV measures
        abyssal clay directly; the continental-terrace fit a deep-ocean caller
        gets by default sits well above it, and each abyssal fit is nearer its
        own measured rows. Densities are comparable directly (both are
        saturated bulk density at the reference water); the velocities are not,
        the paper's being in-situ-corrected absolutes."""
        measured = [('abyssal-plain', 9.53, 1.352),
                    ('abyssal-hill', 9.43, 1.414),
                    ('abyssal-hill', 8.76, 1.344)]
        for environment, phi, rho in measured:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                terrace = grain_size_to_geoacoustics(phi)['density']
                own = grain_size_to_geoacoustics(
                    phi, environment=environment)['density']
            assert abs(own - rho) < abs(terrace - rho), environment
            assert abs(own - rho) < 0.05

    def test_a_density_outside_a_fits_range_is_announced_not_absorbed(self):
        """The inverse has the same duty as the forward direction. Its
        no-solution branch is the one that matters: below the terrace
        quadratic's minimum there is no root at all, which is 45.5 % of the
        Graw grid's ocean cells, so returning the fine end quietly would be the
        flat hold again in a new place. Both ends, and the other side."""
        from uacpy.core.sediment import (grain_size_from_density,
                                         GRAIN_SIZE_ENVIRONMENTS)
        fit = GRAIN_SIZE_ENVIRONMENTS['continental-terrace']
        lo, hi = fit['evaluated_range']
        for phi in (lo, 0.0, 4.0, hi):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                rho = grain_size_to_geoacoustics(float(phi))['density']
                assert grain_size_from_density(rho) == pytest.approx(phi)
            # Silent *about the density*; at -1 ϕ the forward call still
            # reports its attenuation, which is a different statement.
            assert not [w for w in caught if 'reproduces' in str(w.message)]
        for rho, end in ((1.35, hi), (1.20, hi), (2.9, lo)):
            with pytest.warns(UserWarning, match='reproduces'):
                assert grain_size_from_density(rho) == pytest.approx(end)

    def test_the_inverse_names_an_environment_that_covers_the_density(self):
        """1.35 g/cm³ is ordinary abyssal mud — Table IV measures 1.352 for
        abyssal-plain clay — and the terrace relation cannot represent it at
        all. Naming the fit that can is what makes the warning actionable,
        and that fit inverts the same density without one."""
        from uacpy.core.sediment import grain_size_from_density
        with pytest.warns(UserWarning, match='abyssal-plain covers this'):
            grain_size_from_density(1.35)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            phi = grain_size_from_density(1.35, environment='abyssal-plain')
        assert 7.0 <= phi <= 10.0

    def test_an_unknown_environment_and_an_apl_uw_environment_are_refused(self):
        """TR 9407 publishes one set of relations, not one per environment, so
        asking it for an abyssal fit is asking for something that does not
        exist — better refused than silently ignored."""
        with pytest.raises(ConfigurationError, match='unknown environment'):
            grain_size_to_geoacoustics(5.0, environment='abyss')
        with pytest.raises(ConfigurationError, match='has no'):
            grain_size_to_geoacoustics(8.0, model='apl-uw',
                                       environment='abyssal-hill')
        # The default is accepted by both, so nothing moves for a caller who
        # never names one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert (grain_size_to_geoacoustics(4.0, model='apl-uw')
                    == grain_size_to_geoacoustics(
                        4.0, model='apl-uw',
                        environment='continental-terrace'))

    def test_the_velocity_and_density_interval_rests_on_two_documents(self):
        """-1 to 9 ϕ is not one citation. **1 to 9** is Hamilton & Bachman's own
        limit for the (T) regressions (p. 1902); **-1 to 1** is TR 9407, which
        prints those same polynomials as its own coarse branch and declares -1
        its floor. The structure has to keep both halves attributable, so the
        two constants it is built from stay separate."""
        from uacpy.core.sediment import (_HB_T_PHI_RANGE, _APL_UW_PHI_RANGE,
                                         _HB_T_EVALUATED_RANGE)
        assert _HB_T_PHI_RANGE == (1.0, 9.0)            # Hamilton & Bachman
        assert _APL_UW_PHI_RANGE == (-1.0, 9.0)         # TR 9407
        assert _HB_T_EVALUATED_RANGE == (_APL_UW_PHI_RANGE[0],
                                         _HB_T_PHI_RANGE[1])
        for quantity in ('sound_speed', 'density'):
            assert (GRAIN_SIZE_SOURCE_RANGES['hamilton'][quantity]
                    == _HB_T_EVALUATED_RANGE)

    def test_the_attenuation_is_one_function_not_two_copies(self):
        """TR 9407 reproduces Hamilton (1972), so uacpy evaluates it once. Two
        copies used to differ by the tie at a branch join — ``<`` against
        ``<=`` — returning different values from identical coefficients at
        exactly 2.6, 4.5 and 6.0 ϕ. There is nothing left to differ: the two
        models' k_p is the same object, and the join belongs to the branch
        above it, as ``ReadEnvironmentBell.f90:509-518`` reads it."""
        from uacpy.core.sediment import _hamilton_kp
        water = dict(water_sound_speed=1500.0, water_density=1.0)
        for phi in np.linspace(-1.0, 9.0, 1001):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                h = grain_size_to_geoacoustics(float(phi), **water)
                a = grain_size_to_geoacoustics(float(phi), model='apl-uw',
                                               **water)
            # k_p itself: the dB/λ differs through each model's own c.
            assert (h['attenuation'] / h['sound_speed']
                    == pytest.approx(a['attenuation'] / a['sound_speed'],
                                     abs=1e-15)), phi
        # The tie: at a join the value is continuous with the branch *above*
        # and steps away from the branch below, which is AT's reading.
        for join in (2.6, 4.5, 6.0):
            assert _hamilton_kp(join) == pytest.approx(
                _hamilton_kp(join + 1e-9), abs=1e-7)
            assert abs(_hamilton_kp(join) - _hamilton_kp(join - 1e-9)) > 1e-4

    def test_the_apl_uw_branches_are_the_terrace_fit_rescaled(self):
        """TR 9407 p. IV-8: "The density and sound speed ratios agree with
        those of Hamilton and Bachman for -1 <= M_z < 1". They are the same
        polynomials over the published 1528.0 m/s and 1.026 g/cm³, and the
        printed coefficients are *kept* rather than derived because AT
        implements the printed digits and ``'apl-uw'`` exists to reproduce AT.
        This ties the two so a correction to either cannot leave the other
        stale — the residual is the rounding of TR 9407's own 4-to-6 digits."""
        from uacpy.core.sediment import (_apl_density_ratio,
                                         _apl_velocity_ratio, _HB_T_DENSITY,
                                         _HB_T_REF_CW, _HB_T_REF_RHOW,
                                         _HB_T_VELOCITY)
        for mz in np.linspace(-1.0, 0.999, 200):
            rescaled_nu = np.polyval(_HB_T_VELOCITY[::-1], mz) / _HB_T_REF_CW
            rescaled_rho = np.polyval(_HB_T_DENSITY[::-1], mz) / _HB_T_REF_RHOW
            assert _apl_velocity_ratio(mz) == pytest.approx(rescaled_nu,
                                                            abs=2e-5), mz
            assert _apl_density_ratio(mz) == pytest.approx(rescaled_rho,
                                                           abs=7e-5), mz

    @pytest.mark.parametrize('phi', [-1.0, -0.5, 0.0, 0.5, 0.999])
    def test_the_two_models_agree_below_one_phi_because_they_are_one_fit(
            self, phi):
        """TR 9407 p. IV-8: "The density and sound speed ratios agree with
        those of Hamilton and Bachman for -1 <= M_z < 1" — its coarse branch is
        those polynomials over 1528 m/s and 1.026 g/cm³. Given the same water,
        the two models must therefore return the same velocity and density
        there. They agree to 0.03 m/s rather than exactly, which is the
        rounding of TR 9407's own printed 4-to-6-digit coefficients."""
        water = dict(water_sound_speed=1500.0, water_density=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            h = grain_size_to_geoacoustics(phi, model='hamilton', **water)
            a = grain_size_to_geoacoustics(phi, model='apl-uw', **water)
        assert h['sound_speed'] == pytest.approx(a['sound_speed'], abs=0.03)
        assert h['density'] == pytest.approx(a['density'], abs=1e-4)
        assert h['attenuation'] == pytest.approx(a['attenuation'], abs=2e-4)

    @pytest.mark.parametrize('phi, edge', [(float('inf'), 9.0),
                                           (float('-inf'), -1.0)])
    def test_an_infinite_phi_is_reported_like_any_other_clamp(self, phi, edge):
        """±inf is the largest substitution the clamp can make — the apl-uw
        polynomials run to ∓inf there — and it returns the same endpoint the
        finite ϕ = 12 does, which already warns."""
        with pytest.warns(UserWarning, match='clamped'):
            out = grain_size_to_geoacoustics(phi, model='apl-uw')
        assert out == grain_size_to_geoacoustics(edge, model='apl-uw')

    def test_a_nan_phi_is_not_reported_as_a_clamp(self):
        """NaN compares unequal to its own clip, which would make a bare
        ``phi != grain_size_phi`` test claim a clamp that never happened.
        Unlike ±inf it substitutes nothing: it propagates to NaN outputs."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = grain_size_to_geoacoustics(float('nan'), model='apl-uw')
        assert np.isnan(out['sound_speed'])


def test_hamilton_honours_its_own_attenuation_range_to_nine_and_a_half_phi():
    """Hamilton (1972) recommends the ``k_p`` regressions over 0 to 9.5 ϕ,
    while Hamilton & Bachman declare their velocity/density fits to 9. The
    model's ϕ range is the wider of the two: between 9 and 9.5 the quadratics
    are held at 9 while ``k_p`` keeps following its own regression, which turns
    up from 0.080 at 9 to 0.090 at 9.5. That is the case no single verdict can
    describe, so the call reports it per quantity — velocity and density named,
    attenuation not."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        edge = sediment.grain_size_to_geoacoustics(9.0, model='hamilton')
    with pytest.warns(UserWarning) as record:
        clay = sediment.grain_size_to_geoacoustics(9.5, model='hamilton')
    message = str(record[0].message)
    assert 'sound_speed and density' in message
    assert 'attenuation' not in message
    assert clay['sound_speed'] == pytest.approx(edge['sound_speed'])
    assert clay['density'] == pytest.approx(edge['density'])
    assert clay['attenuation'] > edge['attenuation'] * 1.1
    assert GRAIN_SIZE_MODEL_RANGES['hamilton'] == (-1.0, 9.5)


# --------------------------------------------------------------------------
# The same relations, written down twice and published twice.
#
# Two modules evaluate the four relations above for their own reasons --
# :mod:`uacpy.core.sediment` to build a seabed a propagation model can use,
# :mod:`uacpy.sonar.bottom_scattering` to reproduce TR 9407's own scattering
# tables -- and a second compilation, Ainslie's, prints both models' outputs
# independently of either. The classes below hold those two subjects: that the
# package's two copies of a relation have not drifted, and that what they
# compute matches a source neither was written from.
# --------------------------------------------------------------------------

#: The interval both implementations evaluate: TR 9407's own -1 <= Mz <= 9
#: (p. IV-8).
_SHARED_RANGE = (-1.0, 9.0)
#: The joins of Hamilton (1972) Fig. 3's four branches, where a tie-break
#: decides the value and the two implementations could disagree while matching
#: coefficient for coefficient everywhere else.
_BRANCH_JOINS = (2.6, 4.5, 6.0)


class TestCoreAndSonarEvaluateOneRelation:
    """Neither implementation is a copy to be deleted, and they must not drift.

    Each exists to reproduce a different published artefact -- the
    Acoustics-Toolbox ``'G'`` bottom on one side, TR 9407's scattering tables
    on the other -- and coupling them would put a propagation-side edit in the
    path of a scattering-side fidelity test. A failure here is not a bug in
    whichever module was edited last: it means the two have parted, and that
    has to become a decision. It should never be made to pass by loosening a
    tolerance.
    """

    @pytest.mark.parametrize('relation, core_fn, sonar_fn', [
        ('attenuation', _hamilton_kp, _grain_size_alpha_over_f),
        ('speed ratio', _apl_velocity_ratio, _grain_size_speed_ratio),
        ('density ratio', _apl_density_ratio, _grain_size_density_ratio),
    ])
    def test_one_relation_however_many_modules_write_it_down(
            self, relation, core_fn, sonar_fn):
        grid = np.linspace(*_SHARED_RANGE, 4001)
        worst, where = 0.0, None
        for mz in grid:
            gap = abs(core_fn(float(mz)) - sonar_fn(float(mz)))
            if gap > worst:
                worst, where = gap, float(mz)
        assert worst < 1e-12, (
            f"{relation}: uacpy.core.sediment and uacpy.sonar."
            f"bottom_scattering differ by {worst:.3e} at Mz={where}. They "
            f"implement the same published regression; if they must now "
            f"differ, say which artefact each reproduces and record the split "
            f"here rather than widening this bound.")

    def test_the_branch_joins_are_tied_the_same_way_on_both_sides(self):
        """The place they came closest to parting, and did.

        Hamilton's Fig. 3 caption gives the branch ranges as "0 to 2.6 phi",
        "2.6 to 4.5 phi" and so on, sharing each endpoint between two branches
        and settling nothing, so a tie-break is an implementation choice. The
        two sides once made it differently -- ``<=`` in core against ``<`` in
        sonar -- and returned different attenuations at exactly these three phi
        from identical coefficients. Both now read the join as belonging to the
        branch **above** it, which is how ``ReadEnvironmentBell.f90:509-518``
        reads it (``ELSE IF( Mz >= 2.6 .AND. Mz < 4.5 )``) and therefore what
        the Acoustics-Toolbox ``'G'`` bottom does.
        """
        for join in _BRANCH_JOINS:
            assert _hamilton_kp(join) == pytest.approx(
                _grain_size_alpha_over_f(join), abs=1e-12), join
            # Continuous with the branch above, stepping away from the one
            # below: the tie itself, not merely the agreement.
            for evaluate in (_hamilton_kp, _grain_size_alpha_over_f):
                assert evaluate(join) == pytest.approx(evaluate(join + 1e-9),
                                                       abs=1e-7)
                assert abs(evaluate(join) - evaluate(join - 1e-9)) > 1e-4


#: Ainslie, *Principles of Sonar Performance Modelling* (2010) Table 4.17,
#: p. 176 -- "Default HF geo-acoustic parameters (10-100 kHz)", the APL-UW set:
#: M_z, c_HF/c_w, rho_HF/rho_w, alpha_HF (dB/lambda).
_AINSLIE_HF_TABLE = [
    (-1.0, 1.3370, 2.492, 0.91), (-0.5, 1.3067, 2.401, 0.89),
    (0.0, 1.2778, 2.314, 0.87), (0.5, 1.2503, 2.231, 0.87),
    (1.0, 1.2241, 2.151, 0.88), (1.5, 1.1782, 1.845, 0.86),
    (2.0, 1.1396, 1.615, 0.86), (2.5, 1.1073, 1.451, 0.85),
    (3.0, 1.0800, 1.339, 0.92), (3.5, 1.0568, 1.268, 1.00),
    (4.0, 1.0364, 1.224, 1.07), (4.5, 1.0179, 1.195, 1.15),
    (5.0, 0.9999, 1.169, 0.67), (5.5, 0.9885, 1.149, 0.36),
    (6.0, 0.9873, 1.149, 0.20), (6.5, 0.9861, 1.148, 0.16),
    (7.0, 0.9849, 1.147, 0.13), (7.5, 0.9837, 1.147, 0.10),
    (8.0, 0.9824, 1.146, 0.09), (8.5, 0.9812, 1.145, 0.08),
    (9.0, 0.9800, 1.145, 0.08),
]
#: His Table 4.18, p. 178 -- "Default MF geo-acoustic parameters (1-10 kHz)",
#: the bulk set, which is Bachman (1985): a DIFFERENT fit to the same kind of
#: data, published three years after the regressions ``'hamilton'`` evaluates.
_AINSLIE_MF_TABLE = [
    (-1.0, 1.3370, 2.492, 0.91), (-0.5, 1.3067, 2.401, 0.89),
    (0.0, 1.2778, 2.314, 0.87), (0.5, 1.2503, 2.231, 0.87),
    (1.0, 1.2226, 2.162, 0.87), (1.5, 1.1978, 2.086, 0.88),
    (2.0, 1.1743, 2.014, 0.88), (2.5, 1.1522, 1.945, 0.89),
    (3.0, 1.1314, 1.879, 0.96), (3.5, 1.1120, 1.817, 1.05),
    (4.0, 1.0939, 1.758, 1.13), (4.5, 1.0772, 1.702, 1.22),
    (5.0, 1.0619, 1.650, 0.71), (5.5, 1.0479, 1.601, 0.38),
    (6.0, 1.0352, 1.555, 0.21), (6.5, 1.0239, 1.513, 0.17),
    (7.0, 1.0140, 1.474, 0.13), (7.5, 1.0054, 1.439, 0.11),
    (8.0, 0.9982, 1.407, 0.09), (8.5, 0.9923, 1.378, 0.08),
    (9.0, 0.9877, 1.353, 0.08),
]
#: Coarser than 0.81 phi Ainslie's MF branch IS his HF one (his Table 4.18's
#: own definition), so over these grain sizes both models must match the table
#: and each other to the printed digit.
_SHARED_COARSE_BRANCH = (-1.0, -0.5, 0.0, 0.5)


def _as_ratios(phi, model):
    """A model's output as ratios to the seawater it is referenced to."""
    ref_cw, ref_rhow = _MODEL_WATER_REFERENCE[model]
    got = grain_size_to_geoacoustics(phi, model=model)
    return (got['sound_speed'] / ref_cw, got['density'] / ref_rhow,
            got['attenuation'])


class TestBothModelsReproduceAPublishedCompilation:
    """Reproducing the source you were written from is not sufficient.

    A transcription error in a coefficient reproduces itself forever. Ainslie
    Sec. 4.4.1 is an independent compilation of the same physics by a different
    route, and agreeing with it is evidence no amount of self-consistency can
    give.

    Both tables print ratios to the seawater each set was referenced to, so
    each model is divided by its own reference before the comparison. That
    division makes every test here **blind to that constant** -- moving
    ``'hamilton'``'s from 1510 m/s to 1500 leaves them all green, which is what
    a mutation showed. It is pinned on its own below instead, because it is a
    choice uacpy makes and not something these tables constrain.

    What a failure means. On ``'apl-uw'`` a coefficient moved: the agreement
    there is to the last printed digit. On ``'hamilton'`` it means more than
    the tolerance allows of the gap between two published fits, which is a
    judgement to make deliberately -- Bachman's own standard errors are 1.5 %
    on sound speed and 7.5 % on density (Ainslie Eq. 4.95), and the measured
    gaps are 0.15 % and 4.2 %. Widening a tolerance to pass is not a fix.
    """

    @pytest.mark.parametrize('phi, speed, density, alpha', _AINSLIE_HF_TABLE)
    def test_apl_uw_reproduces_the_printed_high_frequency_table(
            self, phi, speed, density, alpha):
        got_speed, got_density, got_alpha = _as_ratios(phi, 'apl-uw')
        assert got_speed == pytest.approx(speed, rel=5e-4)
        assert got_density == pytest.approx(density, rel=5e-4)
        assert got_alpha == pytest.approx(alpha, abs=0.02)

    @pytest.mark.parametrize('phi, speed, density, alpha', _AINSLIE_MF_TABLE)
    def test_hamilton_reproduces_the_printed_bulk_table(
            self, phi, speed, density, alpha):
        got_speed, got_density, got_alpha = _as_ratios(phi, 'hamilton')
        # Two fits of the same quantity, not one fit twice: the tolerances are
        # the measured gaps with a margin, both inside Bachman's own standard
        # errors. alpha's worst gap is 0.021 dB/lambda, at 4 phi.
        assert got_speed == pytest.approx(speed, rel=2e-3)
        assert got_density == pytest.approx(density, rel=0.05)
        assert got_alpha == pytest.approx(alpha, abs=0.03)

    @pytest.mark.parametrize('phi', _SHARED_COARSE_BRANCH)
    def test_the_two_models_share_the_coarse_branch_the_tables_share(
            self, phi):
        """Coarser than 0.81 phi the two models are one relation.

        The comparison is between RATIOS, because that is what the tables print
        and what the two models hold in common; each carries its output back to
        a different seawater (1510 m/s / 1.03 against the Acoustics-Toolbox
        1500 / 1.0), so the absolute values differ by that factor by
        construction. The attenuation is in dB/lambda, proportional to the
        sound speed, so it is compared with the same factor divided out -- and
        then it too is one relation: Hamilton (1972)'s k_p on both sides.
        """
        hamilton_cw = _MODEL_WATER_REFERENCE['hamilton'][0]
        apl_uw_cw = _MODEL_WATER_REFERENCE['apl-uw'][0]
        h_speed, h_density, h_alpha = _as_ratios(phi, 'hamilton')
        a_speed, a_density, a_alpha = _as_ratios(phi, 'apl-uw')
        assert h_speed == pytest.approx(a_speed, rel=1e-3)
        assert h_density == pytest.approx(a_density, rel=1e-3)
        # 5e-5 is the residual of the one approximation between the two
        # routes: 'hamilton' forms its velocity ratio as 1952.5/1528 =
        # 1.277814 where APL-UW prints 1.2778, 1.1e-5 apart.
        assert h_alpha / hamilton_cw == pytest.approx(a_alpha / apl_uw_cw,
                                                     rel=5e-5)

    def test_holding_the_attenuation_flat_below_zero_phi_tracks_the_rise(self):
        """The clamp's shape, not just its value.

        ``'hamilton'`` holds Hamilton (1972)'s ``k_p`` at its 0 phi value below
        0 phi, because that is where his data end -- but alpha in dB/lambda is
        ``k_p`` times the sound speed, which keeps rising. Both of Ainslie's
        tables do the same: alpha goes 0.87 -> 0.91 from 0 to -1 phi, a factor
        1.046. A clamp applied to alpha itself instead of to ``k_p`` would hold
        it at 1.000 and pass every value test above within its tolerance.
        """
        at_zero = grain_size_to_geoacoustics(
            0.0, model='hamilton')['attenuation']
        at_minus_one = grain_size_to_geoacoustics(
            -1.0, model='hamilton')['attenuation']
        assert at_minus_one / at_zero == pytest.approx(0.91 / 0.87, rel=5e-3)

    def test_each_model_carries_its_ratios_back_to_its_own_seawater(self):
        """The reference the tests above divide out, pinned where it shows.

        ``'apl-uw'``'s pair is what the Acoustics Toolbox applies to a ``'G'``
        bottom -- ``alphaR = vr * 1500.0`` and the density ratio used directly
        as g/cm3 (``ReadEnvironmentBell.f90:526`` and ``:531``) -- so a uacpy
        grain-size seabed and a Bellhop one are the same seabed.
        ``'hamilton'``'s is uacpy's in-situ default, which is the point of a
        ratio: Hamilton & Bachman measured at 23 degC and 1 atm, and their
        ratio is meant to be re-applied at the conditions of the site.

        Changing either is allowed. Doing it silently is not: nothing else in
        this class, and nothing in either published table, can see it.
        """
        assert _MODEL_WATER_REFERENCE['apl-uw'] == (1500.0, 1.0)
        assert _MODEL_WATER_REFERENCE['hamilton'] == (1510.0, 1.03)


class TestTheDefaultModelHasOneHome:
    """``model='hamilton'`` is read from one constant, not written out 23 times.

    It used to be a literal at 23 call signatures across ten modules. Nothing
    held them together: a deliberate change of default would have had to find
    all 23, and a half-done one would have left ``fetch_bottom_deck41`` and
    ``fetch_bottom_mars`` converting the same grain size two different ways
    without a word.

    The sweep walks the package by signature rather than grepping, because the
    default has three spellings -- ``model='hamilton'``, ``model: str =
    'hamilton'`` and ``bottom_model='hamilton'`` -- and a grep that knows one
    of them reports "all clear" about the two it cannot see.

    Changing the default is allowed: it is a judgement about which band uacpy
    serves by default, and ``DEFAULT_GRAIN_SIZE_MODEL``'s own docstring makes
    the case for the one it has. Changing it in nine places out of ten is not.
    """

    #: A parameter named ``model`` means a dozen things in this package (a
    #: ``Result`` carries the name of the propagation model that made it). The
    #: grain-size ones are those sitting beside one of these.
    COMPANIONS = ('environment', 'grain_size_phi', 'water_sound_speed',
                  'bottom_model')

    @classmethod
    def _entry_points(cls):
        """(module, qualname, parameter, default) for every grain-size default."""
        found = []
        for module in pkgutil.walk_packages(uacpy.__path__, 'uacpy.'):
            if '.tests' in module.name or 'third_party' in module.name:
                continue
            try:
                imported = importlib.import_module(module.name)
            except Exception:             # optional dependency, driver binary
                continue
            for _, member in inspect.getmembers(imported):
                if inspect.isfunction(member):
                    functions = [member]
                elif inspect.isclass(member):
                    functions = [f for _, f in
                                 inspect.getmembers(member, inspect.isfunction)]
                else:
                    continue
                for function in functions:
                    try:
                        signature = inspect.signature(function)
                    except (ValueError, TypeError):
                        continue
                    names = set(signature.parameters)
                    for name in ('model', 'bottom_model'):
                        parameter = signature.parameters.get(name)
                        if parameter is None:
                            continue
                        if parameter.default is inspect.Parameter.empty:
                            continue
                        if name == 'model' and not (names & set(cls.COMPANIONS)):
                            continue
                        found.append((module.name, function.__qualname__,
                                      name, parameter.default))
        return found

    def test_the_sweep_finds_the_entry_points_it_exists_to_check(self):
        """A sweep that finds nothing passes the test below every time."""
        found = self._entry_points()
        assert len(found) >= 20, f"only {len(found)} entry point(s) found"
        modules = {module for module, _, _, _ in found}
        assert 'uacpy.data.mars' in modules
        assert 'uacpy.data.sediment_db' in modules
        assert 'uacpy.core.sediment' in modules

    def test_every_grain_size_entry_point_reads_the_one_default(self):
        disagreeing = []
        for module, qualname, parameter, default in self._entry_points():
            if default is None:               # "inherit from my caller"
                continue
            if default not in GRAIN_SIZE_MODELS:
                disagreeing.append(
                    f"{module}.{qualname}({parameter}={default!r}) "
                    f"is not a known model")
            elif default != DEFAULT_GRAIN_SIZE_MODEL:
                disagreeing.append(
                    f"{module}.{qualname}({parameter}={default!r})")
        assert not disagreeing, (
            "these entry points do not read DEFAULT_GRAIN_SIZE_MODEL = "
            f"{DEFAULT_GRAIN_SIZE_MODEL!r}:\n  " + "\n  ".join(disagreeing))
