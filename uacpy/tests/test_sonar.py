"""Tests for the ``uacpy.sonar`` package — scattering laws, reverberation,
the sonar equation, and detection theory. Pure-Python; no model binary.
"""

import numpy as np
import pytest
from scipy.stats import norm

from uacpy import sonar
from uacpy.core.exceptions import ConfigurationError


class TestScattering:
    def test_lambert_at_grazing_90_equals_mu(self):
        # sin(90)=1 -> 20*log10(1)=0, so S_b = mu_db.
        assert sonar.lambert_bottom(90.0, mu_db=-27.0) == pytest.approx(-27.0)

    def test_lambert_monotonic_in_angle(self):
        s = sonar.lambert_bottom([5, 20, 45, 80])
        assert np.all(np.diff(s) > 0)

    def test_chapman_harris_finite_and_validates(self):
        assert np.isfinite(sonar.chapman_harris_surface(10.0, 15.0, 3000.0))
        with pytest.raises(ConfigurationError):
            sonar.chapman_harris_surface(10.0, 0.0, 3000.0)

    def test_column_scattering_strength(self):
        assert sonar.column_scattering_strength(-70.0, 100.0) == pytest.approx(-50.0)


class TestReverberation:
    def test_boundary_matches_manual_formula(self):
        r = np.array([1000.0])
        rl = sonar.boundary_reverberation(
            r, 220.0, -40.0, pulse_length_s=0.1,
            horizontal_beamwidth_rad=0.1, sound_speed=1500.0, tl_db=None,
        )
        tl = 20 * np.log10(1000.0)
        cell = 0.1 * 1000.0 * (1500.0 * 0.1 / 2)
        expected = 220.0 - 2 * tl - 40.0 + 10 * np.log10(cell)
        assert rl[0] == pytest.approx(expected)

    def test_volume_grows_faster_than_boundary(self):
        r = np.array([500.0, 5000.0])
        b = sonar.boundary_reverberation(r, 220, -40, pulse_length_s=0.1,
                                         horizontal_beamwidth_rad=0.1)
        v = sonar.volume_reverberation(r, 220, -80, pulse_length_s=0.1,
                                       solid_angle_beamwidth_sr=0.01)
        # Volume cell ~ r^2 vs boundary ~ r, so volume falls off less per decade.
        assert (v[0] - v[1]) < (b[0] - b[1])

    def test_total_is_incoherent_sum(self):
        a, b = np.array([80.0]), np.array([74.0])
        tot = sonar.total_reverberation(a, b)
        expected = 10 * np.log10(10 ** 8.0 + 10 ** 7.4)
        assert tot[0] == pytest.approx(expected)

    def test_bad_pulse_raises(self):
        with pytest.raises(ConfigurationError):
            sonar.boundary_reverberation([100.0], 220, -40, pulse_length_s=0.0,
                                         horizontal_beamwidth_rad=0.1)


class TestSonarEquation:
    def test_echo_level(self):
        assert sonar.echo_level(220, 60, 10) == pytest.approx(220 - 120 + 10)

    def test_passive_signal_excess(self):
        se = sonar.passive_signal_excess(140, 80, 60, directivity_index=15,
                                         detection_threshold=5)
        assert se == pytest.approx(140 - 80 - (60 - 15) - 5)

    def test_active_uses_louder_background(self):
        # Reverb (90) louder than noise-DI (45) -> background dominated by reverb.
        se = sonar.active_signal_excess(
            220, 60, 10, noise_level=60, directivity_index=15,
            reverberation_level=90, detection_threshold=10,
        )
        background = 10 * np.log10(10 ** 4.5 + 10 ** 9.0)
        assert se == pytest.approx(220 - 120 + 10 - background - 10)

    def test_active_requires_a_background(self):
        with pytest.raises(ValueError):
            sonar.active_signal_excess(220, 60, 10)

    def test_figure_of_merit(self):
        assert sonar.figure_of_merit(220, 60, 15, 10) == pytest.approx(220 - 45 - 10)

    def test_detection_range_crossing(self):
        r = np.linspace(0, 100, 11)
        se = 50 - r  # crosses zero at r=50
        assert sonar.detection_range(r, se) == pytest.approx(50.0)

    def test_detection_range_all_positive_is_inf(self):
        r = np.linspace(1, 10, 10)
        assert sonar.detection_range(r, np.ones_like(r)) == np.inf

    def test_detection_range_all_negative_is_nan(self):
        r = np.linspace(1, 10, 10)
        assert np.isnan(sonar.detection_range(r, -np.ones_like(r)))


class TestDetection:
    def test_deflection_matches_normal_quantiles(self):
        d = sonar.deflection_coefficient(0.9, 0.01)
        assert d == pytest.approx(norm.ppf(0.9) - norm.ppf(0.01))

    def test_pod_inverts_deflection(self):
        d = sonar.deflection_coefficient(0.9, 0.01)
        assert float(sonar.probability_of_detection(d, 0.01)) == pytest.approx(0.9)

    def test_roc_monotonic(self):
        pf, pd = sonar.roc_curve(2.0)
        assert np.all(np.diff(pd) >= -1e-9)
        assert pd.max() <= 1.0 and pd.min() >= 0.0

    def test_albersheim_reasonable(self):
        # Single-pulse Pd=0.5, Pf=1e-4 -> ~9.4 dB.
        assert sonar.albersheim_snr(0.5, 1e-4) == pytest.approx(9.4, abs=0.3)

    def test_albersheim_integration_lowers_snr(self):
        assert sonar.albersheim_snr(0.9, 1e-6, 10) < sonar.albersheim_snr(0.9, 1e-6, 1)

    def test_detection_threshold_energy_formula(self):
        dt = sonar.detection_threshold_energy(0.9, 0.01, 100.0, 1.0)
        d = sonar.detection_index(0.9, 0.01)
        assert dt == pytest.approx(5 * np.log10(d * 100.0 / 1.0))

    def test_bad_probability_raises(self):
        with pytest.raises(ConfigurationError):
            sonar.deflection_coefficient(1.0, 0.01)
