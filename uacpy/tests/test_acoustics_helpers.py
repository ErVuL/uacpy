"""Reference-value anchors for the scalar helpers in
:mod:`uacpy.core.acoustics` — seawater sound speed / density, the bubble
calculators, and ``power_to_db``. All analytic, no binary.

Sources per pin: Mackenzie (1981) validity ranges; Fofonoff EOS-80
one-atmosphere check values; Medwin & Clay eq. (8.2.13) (the Minnaert
breathing frequency); APL-UW TR 9407 eqs. 28a/28b; the worked numbers in
docs/guide/environment.md §5.
"""

import warnings

import numpy as np
import pytest

from uacpy.core.acoustics import (
    bubble_resonance,
    bubble_soundspeed,
    bubble_surface_loss,
    density,
    power_to_db,
    soundspeed,
)
from uacpy.core.constants import PRESSURE_FLOOR, REFERENCE_PRESSURE_WATER


class TestMackenzieValidityWarnings:
    """``soundspeed`` warns (core/acoustics.py) whenever an input leaves
    Mackenzie's validated ranges — T ∈ [-2, 30] °C, S ∈ [25, 40] PSU,
    D ∈ [0, 8000] m — and stays silent inside them."""

    @pytest.mark.parametrize('kwargs', [
        dict(temperature=35.0),          # T > 30
        dict(temperature=-5.0),          # T < -2
        dict(salinity=10.0),             # S < 25
        dict(salinity=45.0),             # S > 40
        dict(depth=9000.0),              # D > 8000
        dict(depth=-1.0),                # D < 0
    ])
    def test_out_of_range_input_warns_of_extrapolation(self, kwargs):
        with pytest.warns(UserWarning, match='outside validated range'):
            soundspeed(**kwargs)

    def test_in_range_defaults_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            c = soundspeed()                     # T=27, S=35, D=10
        # The default-point value the environment.md bubble example quotes.
        assert c == pytest.approx(1539.087, abs=1e-3)


class TestSeawaterDensityEOS80CheckValues:
    """``density`` is the EOS-80 one-atmosphere equation; the canonical
    UNESCO check values pin every coefficient group (pure water, S, S^1.5,
    S² terms)."""

    def test_standard_seawater_check_values(self):
        # UNESCO (1983) / Millero & Poisson one-atmosphere check values.
        assert density(25.0, 35.0) == pytest.approx(1023.343, abs=1e-3)
        assert density(0.0, 35.0) == pytest.approx(1028.106, abs=1e-3)
        # Pure-water limit (S = 0) at 5 °C.
        assert density(5.0, 0.0) == pytest.approx(999.96675, abs=1e-4)


class TestBubbleResonance:
    """``bubble_resonance`` is Medwin & Clay eq. (8.2.13) — the Minnaert
    breathing frequency f = (1/2πa)·√(3γp_A/ρ_A)."""

    def test_one_millimetre_surface_bubble(self):
        # (1/2π·1e-3)·√(3·1.4·1.013e5/1022.476) = 3246.6 Hz — the classic
        # ~3.25 kHz·mm product (fresh-water ρ=1000 gives 3283).
        assert bubble_resonance(1e-3) == pytest.approx(3246.56, abs=0.01)

    def test_frequency_scales_inversely_with_radius(self):
        assert bubble_resonance(1e-4) == pytest.approx(
            10.0 * bubble_resonance(1e-3), rel=1e-12)

    def test_depth_raises_frequency_as_sqrt_ambient_pressure(self):
        # p_A = p0 + ρ g z, so f(z)/f(0) = √(p_A(z)/p0): 1.4106 at 10 m.
        rho, g = 1022.476, 9.80665
        expected = np.sqrt((1.013e5 + rho * g * 10.0) / 1.013e5)
        assert (bubble_resonance(1e-3, depth=10.0) / bubble_resonance(1e-3)
                == pytest.approx(expected, rel=1e-12))


class TestBubbleSoundspeed:
    def test_documented_void_fraction_drop(self):
        """environment.md §5: a void fraction of only 1e-6 drops
        ``bubble_soundspeed`` by 15.5 m/s (1539.1 → 1523.6) at its default
        reference — Wood's equation is that sensitive to entrained gas."""
        c0 = soundspeed()
        c_bubbly = bubble_soundspeed(1e-6)
        assert c_bubbly == pytest.approx(1523.557, abs=1e-3)
        assert c0 - c_bubbly == pytest.approx(15.53, abs=0.01)

    def test_zero_void_fraction_recovers_the_water_speed(self):
        assert bubble_soundspeed(0.0) == pytest.approx(soundspeed(),
                                                       rel=1e-12)


class TestBubbleSurfaceLoss:
    """``bubble_surface_loss`` is APL-UW TR 9407 eqs. 28a/28b:
    SBL = 1.26e-3/sinβ · U^1.57 · f_kHz^0.85 for U ≥ 6 m/s, continued
    exponentially below the 6 m/s breaking-wave threshold. Returns an
    amplitude multiplier in (0, 1], angle in radians."""

    def test_reference_value_at_10ms_20khz_normal_incidence(self):
        # a = 1.26e-3·10^1.57·20^0.85 = 0.598 dB → multiplier 0.9335.
        assert bubble_surface_loss(10.0, 20000.0, 0.0) == pytest.approx(
            0.93354, abs=1e-4)

    def test_multiplier_bounded_and_monotonic_in_wind(self):
        m3 = bubble_surface_loss(3.0, 20000.0, 0.0)
        m10 = bubble_surface_loss(10.0, 20000.0, 0.0)
        assert 0.0 < m10 < m3 <= 1.0

    def test_continuous_across_the_6ms_breaking_wave_threshold(self):
        below = bubble_surface_loss(5.999, 20000.0, 0.0)
        at = bubble_surface_loss(6.0, 20000.0, 0.0)
        assert below == pytest.approx(at, abs=1e-4)

    def test_angle_enters_as_one_over_sin_of_the_grazing_angle(self):
        # angle is incidence in radians; β = π/2 − angle, and the dB loss
        # scales exactly as 1/sin β = 1/cos(angle).
        db0 = -20.0 * np.log10(bubble_surface_loss(10.0, 20000.0, 0.0))
        db1 = -20.0 * np.log10(bubble_surface_loss(10.0, 20000.0, 1.0))
        assert db1 / db0 == pytest.approx(1.0 / np.cos(1.0), rel=1e-9)


class TestPowerToDb:
    """``power_to_db`` floors ``power`` at :data:`PRESSURE_FLOOR` before the
    log, so a silent sample yields a finite very negative level, never
    ``-inf`` (DOCUMENTATION.md §14)."""

    def test_zero_power_is_finite_at_the_floor_level(self):
        out = power_to_db(0.0)
        assert np.isfinite(out)
        assert out == pytest.approx(
            10.0 * np.log10(PRESSURE_FLOOR / REFERENCE_PRESSURE_WATER ** 2))
        assert out == pytest.approx(-180.0)      # 1e-30 / (1e-6)² = 1e-18

    def test_reference_power_reads_zero_db(self):
        assert power_to_db(REFERENCE_PRESSURE_WATER ** 2) == pytest.approx(0.0)

    def test_custom_floor_is_honoured(self):
        assert power_to_db(0.0, floor=1e-12) == pytest.approx(
            10.0 * np.log10(1e-12 / REFERENCE_PRESSURE_WATER ** 2))
