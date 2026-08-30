"""Tests for the sediment model: grain size in, geoacoustics out.

uacpy.core.sediment holds the conversion itself and uacpy.data.sediment
the lookups and transect assembly built on it; the former is re-exported
through the latter, so both halves are one subject and live here together.

The conversion uses the Hamilton & Bachman (1982) continental-terrace relations
(density / sound-speed ratios, water-referenced) plus the Hamilton (1980) k_p
attenuation law, summarized in the open-access ESAB supplement.

The model has a validity range in phi, and the clamp at its edge is the subject
of the last class: a value outside the range is substituted, and the warning has
to fire on the substitution rather than on merely reaching the boundary, or the
caller cannot tell a clamped answer from an evaluated one.
"""

import warnings

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core.materials import MATERIALS
from uacpy.core.sediment import _MODEL_RANGE, grain_size_to_geoacoustics
from uacpy.data import sediment


def test_hamilton_table_endpoints():
    # Coarse sand and silty clay reproduce the Hamilton & Bachman (1982) table
    # at the reference seawater (c_w=1510 m/s, rho_w=1.030 g/cm3).
    coarse = sediment.grain_size_to_geoacoustics(0.92)
    assert coarse['sound_speed'] == pytest.approx(1813.5, abs=1.0)
    assert coarse['density'] == pytest.approx(2.034, abs=1e-3)
    fine = sediment.grain_size_to_geoacoustics(8.80)
    assert fine['sound_speed'] == pytest.approx(1494.9, abs=1.0)
    assert fine['density'] == pytest.approx(1.480, abs=1e-3)


def test_velocity_ratio_dips_below_water_for_mud():
    # Fine muds are slower than seawater (velocity ratio < 1).
    fine = sediment.grain_size_to_geoacoustics(8.5, water_sound_speed=1500.0)
    assert fine['sound_speed'] < 1500.0


def test_monotonic_speed_and_density():
    coarse = sediment.grain_size_to_geoacoustics(1.0)
    fine = sediment.grain_size_to_geoacoustics(8.0)
    assert coarse['sound_speed'] > fine['sound_speed']
    assert coarse['density'] > fine['density']


def test_attenuation_peaks_in_sand():
    # Hamilton's k_p attenuation peaks in medium/fine sand, not silt or clay.
    phis = np.linspace(0.5, 8.5, 33)
    alpha = [sediment.grain_size_to_geoacoustics(p)['attenuation'] for p in phis]
    peak_phi = phis[int(np.argmax(alpha))]
    assert 2.0 < peak_phi < 5.0


def test_water_referencing_scales_speed():
    warm = sediment.grain_size_to_geoacoustics(2.5, water_sound_speed=1540.0)
    cold = sediment.grain_size_to_geoacoustics(2.5, water_sound_speed=1480.0)
    assert warm['sound_speed'] > cold['sound_speed']


def test_out_of_range_phi_is_clamped_to_the_table_end():
    # Hamilton is an np.interp lookup, so ϕ past the table returns the end row
    # whether it is clamped first or not — silently, because the clamp has
    # nothing to announce.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        g = sediment.grain_size_to_geoacoustics(12.0)
    assert g['sound_speed'] == pytest.approx(1494.9, abs=1.0)   # → silty clay


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
    # APL-UW TR 9407 high-frequency variant: valid output, attenuation peaks in
    # sand, and (per IV-8) lower density/speed than Hamilton at intermediate Mz.
    a = sediment.grain_size_to_geoacoustics(2.0, model='apl-uw')
    h = sediment.grain_size_to_geoacoustics(2.0, model='hamilton')
    assert 1400 < a['sound_speed'] < 1800
    assert a['density'] < h['density']            # APL-UW runs lower at mid-Mz
    phis = np.linspace(-1, 9, 41)
    alpha = [sediment.grain_size_to_geoacoustics(p, model='apl-uw')['attenuation']
             for p in phis]
    assert 2.0 < phis[int(np.argmax(alpha))] < 5.0
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
    (root / 'sediment' / 'deck41.csv').write_text(
        'latitude,longitude,lithology\n50.001,0.0,gravel\n60.0,0.0,gravel\n')
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
    # 'gravel' maps to ϕ -2.0; the grain-size sample is >1000 km away here.
    sample = fetch_sediment_sample((59.9, 0.0), max_distance_km=50.0)
    assert sample['phi'] == pytest.approx(-2.0)


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
        db = self._cache(tmp_path, monkeypatch)
        assert db.fetch_sediment_sample(point)['dataset'] == expected

    @pytest.mark.parametrize('point, expected', [
        ((-20.0, -140.0), 'deck41'),
        ((30.0, -40.0), 'grainsize'),
    ])
    def test_the_bottom_provenance_cites_that_index(self, point, expected,
                                                    tmp_path, monkeypatch):
        db = self._cache(tmp_path, monkeypatch)
        bottom = db.fetch_bottom_local(point)
        assert bottom.data_sources[0].source.id == expected


class TestGrainSizeClampWarnsOnSubstitutionNotOnCrossing:
    """The clamp to the model's ϕ range is unconditional; the warning used to
    be gated on a ±1 ϕ deadband. That gate was wrong in both directions —
    silent for ``apl-uw`` where the clamp moved the answer by up to 47 m/s,
    and reserved for ``hamilton`` where it can never move anything."""

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

    @pytest.mark.parametrize('phi', [8.8001, 9.3, 9.799, 9.801, 0.42, -0.081,
                                     20.0, -20.0])
    def test_hamilton_never_warns_because_its_clamp_is_a_no_op(self, phi):
        """``_hamilton_geoacoustics`` is an ``np.interp`` lookup, which already
        holds the end rows flat past the table. Clamping ϕ first changes
        nothing, so a warning there would be reporting a substitution that did
        not happen."""
        lo, hi = _MODEL_RANGE['hamilton']
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = grain_size_to_geoacoustics(phi, model='hamilton')
        edge = grain_size_to_geoacoustics(
            float(np.clip(phi, lo, hi)), model='hamilton')
        assert out == edge

    def test_a_non_finite_phi_is_not_reported_as_a_clamp(self):
        """NaN compares unequal to its own clip, which would make a bare
        ``phi != grain_size_phi`` test claim a clamp that never happened."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = grain_size_to_geoacoustics(float('nan'), model='apl-uw')
        assert np.isnan(out['sound_speed'])
