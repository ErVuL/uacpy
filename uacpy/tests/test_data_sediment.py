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
    coarse = sediment.grain_size_to_geoacoustics(0.5)
    assert coarse['sound_speed'] == pytest.approx(1813.5, abs=1.0)
    assert coarse['density'] == pytest.approx(2.034, abs=1e-3)
    fine = sediment.grain_size_to_geoacoustics(8.5)
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
