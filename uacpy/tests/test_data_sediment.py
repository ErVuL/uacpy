"""Tests for grain-size → geoacoustic conversion (uacpy.data.sediment).

The conversion uses the Hamilton & Bachman (1982) continental-terrace relations
(density / sound-speed ratios, water-referenced) plus the Hamilton (1980) k_p
attenuation law, summarized in the open-access ESAB supplement.
"""

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.materials import MATERIALS
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


def test_out_of_range_warns_and_clamps():
    with pytest.warns(UserWarning, match='well outside'):
        g = sediment.grain_size_to_geoacoustics(12.0)
    assert g['sound_speed'] == pytest.approx(1494.9, abs=1.0)   # → silty clay


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


def test_auto_still_splits_on_a_real_sediment_change():
    """The collapse must not merge distinct sediments (ϕ 3 → ϕ 7)."""
    bottom = sediment.range_dependent_bottom_along(
        _phi_bottom(lambda la: 3.0 if la < 40.5 else 7.0,
                    lambda la, lo: 1480.0 + 40.0 * (la - 40.0)),
        (40.0, -30.0), (41.0, -30.0), 'auto', source_label='test',
        max_points=200)
    assert len(bottom.columns) == 2
    assert (bottom.columns[0].halfspace.sound_speed
            > bottom.columns[1].halfspace.sound_speed)


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
    assert [c.halfspace.sound_speed for c in bottom.columns] == [1600.0, 2200.0]
