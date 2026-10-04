"""Tests for the ``uacpy.sonar`` package — scattering laws, reverberation,
the sonar equation, and detection theory. Pure-Python; no model binary.

The last two classes are the positivity guards. Every ``x < 0`` / ``x <= 0``
rejection in this package once let NaN through, because every comparison
against NaN is False; each is now written as the negation of the admissible
condition, and each guard's legitimate zero — a 0 deg grazing angle, a
zero-area range cell — is pinned alongside it so closing the guard cannot
close those too.
"""

import warnings

import numpy as np
from uacpy.core.units import ms_to_knots
import pytest
from scipy.stats import norm

from uacpy import sonar
from uacpy.core.exceptions import ConfigurationError
from uacpy.sonar import (APL_UW_SEDIMENTS, BottomParameters,
                         apl_uw_bottom_backscatter, apl_uw_bottom_loss,
                         apl_uw_surface_backscatter)
from uacpy.sonar.bottom_scattering import TABLE_WATER_SOUND_SPEED
from uacpy.sonar import target_strength as ts
from uacpy.sonar.detection import (detection_threshold_energy,
                                   probability_of_detection, roc_curve)
from uacpy.sonar.reverberation import (boundary_reverberation,
                                       volume_reverberation)
from uacpy.sonar.scattering import (chapman_harris_surface,
                                    column_scattering_strength)
from uacpy.tests.conftest import warning_messages
from uacpy.tests.conftest import recorded_warnings

NAN = float('nan')
INF = float('inf')


class TestScattering:
    def test_lambert_at_grazing_90_equals_mu(self):
        # sin(90)=1 -> 20*log10(1)=0, so S_b = mu_dB.
        assert sonar.lambert_bottom(90.0, mu_dB=-27.0) == pytest.approx(-27.0)

    def test_lambert_mu_dB_constant(self):
        # Mackenzie (1961) measured 10*log10(mu) constant at -27 dB for both
        # 530 and 1030 Hz (Etter eq. 9.6); the module exports it and
        # lambert_bottom defaults to it.
        assert sonar.LAMBERT_MU_DB == -27.0
        assert sonar.scattering.LAMBERT_MU_DB == -27.0
        assert sonar.lambert_bottom(90.0) == pytest.approx(sonar.LAMBERT_MU_DB)

    def test_lambert_monotonic_in_angle(self):
        s = sonar.lambert_bottom([5, 20, 45, 80])
        assert np.all(np.diff(s) > 0)

    def test_chapman_harris_finite_and_validates(self):
        assert np.isfinite(sonar.chapman_harris_surface(frequency=3000.0,
                                                        grazing_deg=10.0, wind_speed_kn=15.0))
        with pytest.raises(ConfigurationError,
                           match='wind_speed_kn and frequency must be > 0'):
            sonar.chapman_harris_surface(frequency=3000.0,
                                         grazing_deg=10.0, wind_speed_kn=0.0)

    def test_column_scattering_strength(self):
        assert sonar.column_scattering_strength(-70.0, 100.0) == pytest.approx(-50.0)


class TestReverberation:
    def test_boundary_matches_manual_formula(self):
        r = np.array([1000.0])
        rl = sonar.boundary_reverberation(
            r, 220.0, -40.0, pulse_length_s=0.1,
            horizontal_beamwidth_rad=0.1, sound_speed=1500.0, tl_dB=None,
        )
        tl = 20 * np.log10(1000.0)
        cell = 0.1 * 1000.0 * (1500.0 * 0.1 / 2)
        expected = 220.0 - 2 * tl - 40.0 + 10 * np.log10(cell)
        assert rl[0] == pytest.approx(expected)
        # Independent hand value (TL=60, cell=7500 m², 10log10(7500)=38.751):
        # 220 - 120 - 40 + 38.751 = 98.751 dB — anchors the formula, not just
        # re-derives it.
        assert rl[0] == pytest.approx(98.751, abs=0.01)

    def test_tl_dB_callable_matches_precomputed(self):
        # tl_dB accepts a callable r -> TL(r); the equation evaluates it on
        # the range grid, matching the same TL passed as a precomputed array.
        r = np.array([500.0, 1000.0, 2000.0, 4000.0])
        tl_fn = lambda rr: 15.0 * np.log10(rr)  # noqa: E731
        b_call = sonar.boundary_reverberation(
            r, 200.0, -27.0, pulse_length_s=0.01,
            horizontal_beamwidth_rad=0.1, tl_dB=tl_fn,
        )
        b_arr = sonar.boundary_reverberation(
            r, 200.0, -27.0, pulse_length_s=0.01,
            horizontal_beamwidth_rad=0.1, tl_dB=15.0 * np.log10(r),
        )
        np.testing.assert_allclose(b_call, b_arr)
        v_call = sonar.volume_reverberation(
            r, 200.0, -70.0, pulse_length_s=0.01,
            solid_angle_beamwidth_sr=0.01, tl_dB=tl_fn,
        )
        v_arr = sonar.volume_reverberation(
            r, 200.0, -70.0, pulse_length_s=0.01,
            solid_angle_beamwidth_sr=0.01, tl_dB=15.0 * np.log10(r),
        )
        np.testing.assert_allclose(v_call, v_arr)
        # Spherical-spreading callable reproduces the tl_dB=None default.
        b_sph = sonar.boundary_reverberation(
            r, 200.0, -27.0, pulse_length_s=0.01,
            horizontal_beamwidth_rad=0.1,
            tl_dB=lambda rr: 20.0 * np.log10(rr),
        )
        b_none = sonar.boundary_reverberation(
            r, 200.0, -27.0, pulse_length_s=0.01,
            horizontal_beamwidth_rad=0.1, tl_dB=None,
        )
        np.testing.assert_allclose(b_sph, b_none)

    def test_volume_decays_slower_than_boundary(self):
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
        # Independent hand value: 10log10(1e8 + 2.5119e7) = 80.973 dB.
        assert tot[0] == pytest.approx(80.973, abs=0.01)

    def test_total_survives_levels_that_overflow_a_linear_sum(self):
        """``10**(x/10)`` is inf past ~3080 dB; the sum goes through
        logaddexp, as WenzNoise's does, and stays exact: two equal levels
        add 10·log10(2)."""
        tot = sonar.total_reverberation(np.array([4000.0]),
                                        np.array([4000.0]))
        assert tot[0] == pytest.approx(4000.0 + 10.0 * np.log10(2.0))

    def test_bottom_scattering_is_an_attribute_but_not_listed(self):
        """``apl_uw_bottom_backscatter``'s docstring names
        ``bottom_scattering.TABLE_WATER_SOUND_SPEED``; the submodule is
        reachable as ``sonar.bottom_scattering`` as written, while
        ``__all__`` lists functions, classes and constants only."""
        assert 'bottom_scattering' not in sonar.__all__
        assert sonar.bottom_scattering.TABLE_WATER_SOUND_SPEED > 0

    def test_bad_pulse_raises(self):
        with pytest.raises(
                ConfigurationError,
                match='pulse_length_s and horizontal_beamwidth_rad must be > 0'):
            sonar.boundary_reverberation([100.0], 220, -40, pulse_length_s=0.0,
                                         horizontal_beamwidth_rad=0.1)


class TestSonarEquation:
    def test_echo_level(self):
        assert sonar.echo_level(220, 60, 10) == pytest.approx(220 - 120 + 10)

    def test_detection_range_ignores_no_data_nan(self):
        """NaN marks a cell the propagation model never filled, not a cell
        where the target is undetectable. Treating NaN as 'SE < 0' returned
        'never detectable' for a target detectable to 9 km."""
        r = np.array([1000., 3000., 5000., 7000., 9000., 11000.])
        se = np.array([10.0, np.nan, 6.0, np.nan, 1.0, -3.0])
        assert sonar.detection_range(r, signal_excess_dB=se) == pytest.approx(9500.0)
        # an all-NaN row is genuinely unknown -> nan, not inf
        assert np.isnan(sonar.detection_range(r, signal_excess_dB=np.full(6, np.nan)))

    def test_passive_signal_excess(self):
        se = sonar.passive_signal_excess(140, 80, 60, directivity_index_dB=15,
                                         detection_threshold_dB=5)
        assert se == pytest.approx(140 - 80 - (60 - 15) - 5)

    def test_active_uses_louder_background(self):
        # Reverb (90) louder than noise-DI (45) -> background dominated by reverb.
        se = sonar.active_signal_excess(
            220, 60, 10, noise_level_dB=60, directivity_index_dB=15,
            reverberation_level_dB=90, detection_threshold_dB=10,
        )
        background = 10 * np.log10(10 ** 4.5 + 10 ** 9.0)
        assert se == pytest.approx(220 - 120 + 10 - background - 10)
        # Independent hand value: reverb (90) swamps noise-DI (45), so
        # background ≈ 90.0 dB → SE ≈ 220-120+10-90-10 = 10.0 dB.
        assert se == pytest.approx(10.0, abs=0.01)

    def test_active_requires_a_background(self):
        with pytest.raises(
                ConfigurationError,
                match='provide noise_level_dB and/or reverberation_level_dB'):
            sonar.active_signal_excess(220, 60, 10, detection_threshold_dB=0.0)

    def test_figure_of_merit(self):
        assert sonar.figure_of_merit(220, 60, 15,
                                     detection_threshold_dB=10) == pytest.approx(220 - 45 - 10)

    def test_detection_range_crossing(self):
        r = np.linspace(0, 100, 11)
        se = 50 - r  # crosses zero at r=50
        assert sonar.detection_range(r, signal_excess_dB=se) == pytest.approx(50.0)

    def test_detection_range_all_positive_is_inf(self):
        r = np.linspace(1, 10, 10)
        assert sonar.detection_range(r, signal_excess_dB=np.ones_like(r)) == np.inf

    def test_detection_range_all_negative_is_nan(self):
        r = np.linspace(1, 10, 10)
        assert np.isnan(sonar.detection_range(r, signal_excess_dB=-np.ones_like(r)))

    def test_detection_range_first_mode_reports_first_loss(self):
        # Direct path to 1.5, shadow, convergence zone 2.5..3.5: 'first' is
        # the first-loss range, 'outermost' the far edge of the zone.
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        se = np.array([5.0, 1.0, -1.0, 1.0, -1.0])
        assert sonar.detection_range(r, signal_excess_dB=se, crossing='first') == pytest.approx(1.5)
        assert sonar.detection_range(r, signal_excess_dB=se) == pytest.approx(3.5)

    def test_detection_range_first_crossing_edges(self):
        r = np.linspace(1, 10, 10)
        assert sonar.detection_range(r, signal_excess_dB=np.ones_like(r), crossing='first') == np.inf
        # Shadow at the nearest range: no first-loss range from the source.
        se = np.array([-1.0, 1, 1, -1, -1, -1, -1, -1, -1, -1])
        assert np.isnan(sonar.detection_range(r, signal_excess_dB=se, crossing='first'))
        with pytest.raises(ConfigurationError, match="crossing must be"):
            sonar.detection_range(r, signal_excess_dB=se, crossing='last')

    def test_detection_annuli_lists_every_interval(self):
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        se = np.array([5.0, 1.0, -1.0, 1.0, -1.0, 2.0])
        got = sonar.detection_annuli(r, signal_excess_dB=se)
        assert got == [pytest.approx((0.0, 1.5)), pytest.approx((2.5, 3.5)),
                       pytest.approx((4 + 1 / 3, 5.0))]
        assert sonar.detection_annuli(r, signal_excess_dB=-np.ones_like(r)) == []

    def test_detection_range_outermost_of_multiple_crossings(self):
        r = np.array([0.0, 1.0, 2.0, 3.0])
        se = np.array([5.0, -1.0, 1.0, -1.0])  # down-crossing between r=2 and r=3
        assert sonar.detection_range(r, signal_excess_dB=se) == pytest.approx(2.5)

    def test_detection_range_does_not_interpolate_across_a_no_data_hole(self):
        # The + and - samples bracket an unfilled (NaN) cell, so the crossing
        # lies somewhere the model never computed; the answer is the last
        # positive sample's range, not a sub-cell position inside the hole.
        r = np.array([0.0, 1000.0, 2000.0])
        se = np.array([5.0, np.nan, -5.0])
        assert sonar.detection_range(r, signal_excess_dB=se) == pytest.approx(0.0)

    def test_detection_range_interpolates_between_adjacent_samples(self):
        r = np.array([0.0, 1000.0])
        se = np.array([5.0, -5.0])
        assert sonar.detection_range(r, signal_excess_dB=se) == pytest.approx(500.0)


class TestFarEdgeRecoveryUnderTrailingNoData:
    """The far edge is the last range WITH DATA, not ``ranges[-1]``.

    ``detection_range`` masks the no-data cells before it looks for the far
    edge: SE >= 0 at the outermost FILLED cell puts the crossing beyond the
    grid (inf), and the warning names that cell's range.
    """

    @staticmethod
    def _recovery_with_trailing_no_data(n_empty):
        """A 20 km grid at 1 km spacing whose last ``n_empty`` cells are NaN.

        SE starts positive, dips negative, and is back above zero across the
        eight cells that end at the outermost filled one, so every row takes
        ``detection_range``'s far-edge-recovery branch.
        """
        r = np.arange(0.0, 20001.0, 1000.0)
        se = np.full(r.size, -1.0)
        se[0] = 5.0
        se[-(n_empty + 8):-n_empty] = 2.0
        se[-n_empty:] = np.nan
        return r, se

    def test_far_edge_recovery_names_the_last_range_with_data(self):
        r, se = self._recovery_with_trailing_no_data(3)
        msgs = warning_messages(lambda: sonar.detection_range(r, signal_excess_dB=se), _SHADOW)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert sonar.detection_range(r, signal_excess_dB=se) == np.inf
        assert r[-1] == 20000.0
        assert len(msgs) == 1 and '(17000 m)' in msgs[0]

    def test_by_depth_returns_each_rows_own_last_range_with_data(self):
        """Each row masks its own no-data cells, so "the last range with data"
        is a per-row quantity and no single number can stand for the field."""
        from uacpy.core.results import Field
        r, shallow = self._recovery_with_trailing_no_data(3)
        _, deep = self._recovery_with_trailing_no_data(8)
        field = Field(
            data=np.vstack([shallow, deep]),
            coords={'depth': np.array([10.0, 20.0]), 'range': r},
            model='Bellhop', kind='signal_excess',
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            depths, ranges = sonar.detection_ranges_by_depth(field)
        np.testing.assert_array_equal(depths, [10.0, 20.0])
        assert np.all(ranges == np.inf)


class TestDetection:
    def test_deflection_matches_normal_quantiles(self):
        d = sonar.deflection_coefficient(0.9, 0.01)
        assert d == pytest.approx(norm.ppf(0.9) - norm.ppf(0.01))

    def test_pod_inverts_deflection(self):
        d = sonar.deflection_coefficient(0.9, 0.01)
        assert float(sonar.probability_of_detection(d, 0.01)) == pytest.approx(0.9)

    def test_a_detection_index_goes_back_through_its_square_root(self):
        """detection_index returns d = (d')^2; probability_of_detection takes
        d'. Fed d itself (36.4 at 0.9, 1e-6) it answers P_D = 1."""
        d = sonar.detection_index(0.9, 1e-6)
        assert float(sonar.probability_of_detection(np.sqrt(d), 1e-6)) == \
            pytest.approx(0.9)
        assert float(sonar.probability_of_detection(d, 1e-6)) == \
            pytest.approx(1.0)

    def test_roc_monotonic(self):
        pf, pd = sonar.roc_curve(2.0)
        assert np.all(np.diff(pd) >= -1e-9)
        assert pd.max() <= 1.0 and pd.min() >= 0.0

    def test_roc_curve_default_pf_grid(self):
        # Default grid: logspace(-6, log10(0.99), 200) — 200 points from
        # 1e-6 up to 0.99 (the documented "[1e-6, ~1]"), log-spaced so
        # adjacent-point ratios are constant.
        pf, pd = sonar.roc_curve(2.0)
        assert pf.shape == pd.shape == (200,)
        assert pf[0] == pytest.approx(1e-6, rel=1e-12)
        assert pf[-1] == pytest.approx(0.99, rel=1e-12)
        ratios = pf[1:] / pf[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-10)
        # n_points sets the grid length; endpoints stay pinned.
        pf7, _ = sonar.roc_curve(2.0, n_points=7)
        assert pf7.shape == (7,)
        assert pf7[0] == pytest.approx(1e-6, rel=1e-12)
        assert pf7[-1] == pytest.approx(0.99, rel=1e-12)

    def test_pf_array_matches_scalar_calls(self):
        # probability_of_detection broadcasts over an array of P_F values,
        # returning one P_D per element, equal to the scalar-call results.
        pf = np.array([1e-6, 1e-4, 1e-2, 0.1, 0.5])
        pd = sonar.probability_of_detection(2.0, pf)
        assert pd.shape == pf.shape
        scalars = [float(sonar.probability_of_detection(2.0, p)) for p in pf]
        np.testing.assert_allclose(pd, scalars, rtol=0, atol=1e-15)
        # Higher tolerated P_F -> higher P_D at fixed deflection.
        assert np.all(np.diff(pd) > 0)

    def test_albersheim_reasonable(self):
        # Albersheim's own eq. (1) at N=1, Pd=0.5, Pf=1e-4 evaluates to
        # 9.396 dB, so 9.4 is that value rounded to 1 dp and abs=0.3 is slack
        # on the anchor — NOT the +/-0.2 dB accuracy of the approximation
        # against Robertson, which does not enter a self-consistency check.
        # (Pd, Pf) sit inside the stated validity box (Pd 0.1-0.9,
        # Pf 1e-7..1e-3, N 1..8096; Richards 2014).
        assert sonar.albersheim_snr(0.5, 1e-4) == pytest.approx(9.4, abs=0.3)
        # The exact eq. (1) value: A = ln(0.62/1e-4), B = 0, so
        # SNR = (6.2 + 4.54/sqrt(1.44))*log10(A) = 9.3956 dB.
        assert sonar.albersheim_snr(0.5, 1e-4) == pytest.approx(9.3956,
                                                                abs=1e-3)

    def test_albersheim_matches_closed_form_across_validity_box(self):
        # Abraham eq. (2.85): SNR = -5*log10(N)
        # + (6.2 + 4.54/sqrt(N+0.44))*log10(A + 0.12*A*B + 1.7*B) with
        # A = ln(0.62/Pf), B = ln(Pd/(1-Pd)). Checked at the corners of the
        # Tufts & Cann accuracy box (Pd 0.3-0.95, Pf 1e-8..1e-4, N 1..16;
        # Abraham §2.3.5.2) and at the page's (0.5, 1e-4) operating point.
        for pd, pf, n in [(0.3, 1e-8, 1), (0.95, 1e-4, 1), (0.3, 1e-4, 16),
                          (0.95, 1e-8, 16), (0.5, 1e-4, 1)]:
            a = np.log(0.62 / pf)
            b = np.log(pd / (1.0 - pd))
            ref = (-5.0 * np.log10(n)
                   + (6.2 + 4.54 / np.sqrt(n + 0.44))
                   * np.log10(a + 0.12 * a * b + 1.7 * b))
            assert sonar.albersheim_snr(pd, pf, n) == pytest.approx(ref)

    def test_albersheim_integration_lowers_snr(self):
        assert sonar.albersheim_snr(0.9, 1e-6, 10) < sonar.albersheim_snr(0.9, 1e-6, 1)

    def test_detection_threshold_energy_formula(self):
        dt = sonar.detection_threshold_energy(0.9, 0.01, 100.0, 1.0,
                                              exact=False)
        d = sonar.detection_index(0.9, 0.01)
        # DT = 5*log10(d / (w*t)) (Urick energy detector); anchor to the value.
        assert dt == pytest.approx(5 * np.log10(d / (100.0 * 1.0)))

    def test_detection_threshold_documented_anchors(self):
        # The guide's two budgets (Abraham 9.2.3.1, DT = 5*log10(d/(w*t))):
        # d = (Phi^-1(0.5) - Phi^-1(1e-4))^2 = 13.831, so
        # w*t = 500 -> DT = -7.7906 dB and w*t = 50 -> DT = -2.7906 dB,
        # the doc's -7.79 / -2.79 dB, exactly 5 dB (one decade of w*t) apart.
        d = (norm.ppf(0.5) - norm.ppf(1e-4)) ** 2
        dt_passive = sonar.detection_threshold_energy(
            0.5, 1e-4, bandwidth_hz=50.0, integration_time_s=10.0, exact=False)
        dt_active = sonar.detection_threshold_energy(
            0.5, 1e-4, bandwidth_hz=100.0, integration_time_s=0.5, exact=False)
        assert dt_passive == pytest.approx(5.0 * np.log10(d / 500.0))
        assert dt_active == pytest.approx(5.0 * np.log10(d / 50.0))
        assert dt_passive == pytest.approx(-7.7906, abs=5e-4)
        assert dt_active == pytest.approx(-2.7906, abs=5e-4)
        assert dt_active - dt_passive == pytest.approx(5.0, abs=1e-9)

    def test_the_default_threshold_is_the_exact_one_at_the_guide_anchors(self):
        """The guide's DT_PASSIVE / DT_ACTIVE at (0.5, 1e-4) and w*t = 500 /
        50: 10*log10(S), S = Ginv(1-Pf; M)/Ginv(1-Pd; M) - 1 (Abraham 2.76),
        -7.5519 and -2.0482 dB, 0.24 and 0.74 dB above the large-M form."""
        from scipy.stats import gamma
        for (bw, t, m, want) in ((50.0, 10.0, 500.0, -7.5519),
                                 (100.0, 0.5, 50.0, -2.0482)):
            got = sonar.detection_threshold_energy(0.5, 1e-4, bandwidth_hz=bw,
                                                   integration_time_s=t)
            assert got == pytest.approx(
                10 * np.log10(gamma.isf(1e-4, m) / gamma.isf(0.5, m) - 1),
                abs=1e-12)
            assert got == pytest.approx(want, abs=5e-5)

    def test_detection_threshold_falls_with_time_bandwidth(self):
        # Incoherent integration lowers the required SNR: 5 dB per decade of w*t
        # (Abraham §9.2). DT must DECREASE as bandwidth or integration time grow.
        base = sonar.detection_threshold_energy(0.9, 0.01, 100.0, 1.0, exact=False)
        more_bw = sonar.detection_threshold_energy(0.9, 0.01, 1000.0, 1.0, exact=False)
        more_t = sonar.detection_threshold_energy(0.9, 0.01, 100.0, 10.0, exact=False)
        assert more_bw < base and more_t < base
        assert more_bw == pytest.approx(base - 5.0, abs=1e-9)  # one decade of w
        assert more_t == pytest.approx(base - 5.0, abs=1e-9)   # one decade of t

    def test_bad_probability_raises(self):
        with pytest.raises(ConfigurationError,
                           match=r'pd must be in \(0, 1\)'):
            sonar.deflection_coefficient(1.0, 0.01)

    @pytest.mark.parametrize('call', [
        lambda pd, pf: sonar.deflection_coefficient(pd, pf),
        lambda pd, pf: sonar.detection_index(pd, pf),
        lambda pd, pf: sonar.detection_threshold_energy(pd, pf, 100.0, 1.0),
        lambda pd, pf: sonar.albersheim_snr(pd, pf),
    ])
    def test_an_operating_point_at_or_below_chance_is_refused(self, call):
        # Swapped (pd, pf) returned -9.30 dB, exactly the intended DT,
        # because the detection index squares the deflection.
        for pd, pf in ((1e-5, 1e-3), (1e-3, 1e-3)):
            with pytest.raises(ConfigurationError, match='pd must exceed pf'):
                call(pd, pf)
        assert np.isfinite(call(0.5, 0.49))


class TestSignalExcessField:
    @staticmethod
    def _tl_field(complex_data=False):
        from uacpy.core.results import Field
        depths = np.linspace(0.0, 100.0, 5)
        ranges = np.linspace(1000.0, 9000.0, 7)
        tl = 20.0 * np.log10(ranges)[None, :] + 0.1 * depths[:, None]
        if complex_data:
            data = 10.0 ** (-tl / 20.0) * np.exp(1j * 0.3)
        else:
            data = tl
        return Field(
            data=data,
            coords={'depth': depths, 'range': ranges},
            model='Bellhop',
            frequencies=2000.0,
        ), tl

    def test_passive_matches_scalar_formula(self):
        field, tl = self._tl_field()
        se = sonar.passive_signal_excess_field(
            field, source_level_dB=140.0, noise_level_dB=60.0,
            directivity_index_dB=15.0, detection_threshold_dB=3.0,
        )
        expected = sonar.passive_signal_excess(
            140.0, tl, 60.0, directivity_index_dB=15.0, detection_threshold_dB=3.0,
        )
        np.testing.assert_allclose(se.data, expected)
        assert list(se.coords) == ['depth', 'range']
        np.testing.assert_array_equal(se.coords['range'], field.coords['range'])
        assert se.model == 'Bellhop'
        assert se.sonar_budget['mode'] == 'passive'

    def test_complex_pressure_field_converts_via_tl(self):
        field, tl = self._tl_field(complex_data=True)
        se = sonar.passive_signal_excess_field(
            field, source_level_dB=140.0, noise_level_dB=60.0, detection_threshold_dB=0.0
        )
        expected = 140.0 - tl - 60.0
        np.testing.assert_allclose(se.data, expected, atol=1e-9)

    @staticmethod
    def _surface_row_field():
        """Complex pressure whose first receiver row sits on the
        pressure-release surface: exact zeros there, TL 60-80 dB below."""
        from uacpy.core.results import Field
        field, tl = TestSignalExcessField._tl_field(complex_data=True)
        data = field.data.copy()
        data[0, :] = 0.0
        return Field(data=data, coords=field.coords, model='Scooter',
                     frequencies=2000.0), tl

    @pytest.mark.parametrize('budget', ['passive', 'active'])
    def test_a_cell_no_energy_reached_keeps_the_marker(self, budget):
        from uacpy.core.acoustics.levels import no_energy_mask
        from uacpy.core.constants import NO_ENERGY_DB
        field, tl = self._surface_row_field()
        if budget == 'passive':
            se = sonar.passive_signal_excess_field(
                field, source_level_dB=180.0, noise_level_dB=60.0,
                detection_threshold_dB=0.0)
            expected = 180.0 - tl - 60.0
        else:
            se = sonar.active_signal_excess_field(
                field, source_level_dB=220.0, target_strength_dB=10.0,
                noise_level_dB=60.0, detection_threshold_dB=0.0)
            expected = 220.0 - 2.0 * tl + 10.0 - 60.0
        np.testing.assert_array_equal(se.data[0], -NO_ENERGY_DB)
        assert no_energy_mask(se.data[0]).all()
        np.testing.assert_allclose(se.data[1:], expected[1:], atol=1e-9)
        assert not no_energy_mask(se.data[1:]).any()

    def test_a_two_way_deep_null_is_not_read_as_the_marker(self):
        # 350 dB one way is a real (round-off) null, kept by the mask; two
        # ways it is 700 dB, which the marker test on the product would take.
        from uacpy.core.acoustics.levels import no_energy_mask
        from uacpy.core.results import Field
        field, tl = self._tl_field()
        tl = tl.copy()
        tl[0, 0] = 350.0
        field = Field(data=tl, coords=field.coords, frequencies=2000.0)
        se = sonar.active_signal_excess_field(
            field, source_level_dB=220.0, target_strength_dB=10.0,
            noise_level_dB=60.0, detection_threshold_dB=0.0)
        assert se.data[0, 0] == pytest.approx(220.0 - 700.0 + 10.0 - 60.0)
        assert not no_energy_mask(se.data).any()

    def test_the_excess_map_window_ignores_the_marker_cells(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from uacpy.visualization.plots import plot_signal_excess
        field, tl = self._surface_row_field()
        se = sonar.passive_signal_excess_field(
            field, source_level_dB=140.0, noise_level_dB=60.0,
            detection_threshold_dB=0.0)
        fig, ax = plot_signal_excess(se)
        real = np.abs(140.0 - tl[1:] - 60.0).max()
        mesh = (ax.images or ax.collections)[0]
        assert mesh.get_clim() == pytest.approx((-real, real))
        plt.close(fig)

    def test_active_per_range_reverberation_broadcast(self):
        field, tl = self._tl_field()
        rl = np.linspace(90.0, 70.0, field.coords['range'].size)
        se = sonar.active_signal_excess_field(
            field, source_level_dB=220.0, target_strength_dB=10.0,
            noise_level_dB=60.0, directivity_index_dB=15.0,
            reverberation_level_dB=rl, detection_threshold_dB=3.0,
        )
        expected = sonar.active_signal_excess(
            220.0, tl, 10.0, noise_level_dB=60.0, directivity_index_dB=15.0,
            reverberation_level_dB=rl[None, :], detection_threshold_dB=3.0,
        )
        np.testing.assert_allclose(se.data, expected)
        assert se.sonar_budget['mode'] == 'active'

    def test_the_recorded_budget_is_the_sonar_budget_dict(self):
        # Every budget records all nine terms, None for one not supplied, as
        # SonarBudget.to_dict() writes them.
        terms = {'mode', 'source_level_dB', 'detection_threshold_dB',
                 'noise_level_dB', 'target_strength_dB',
                 'reverberation_level_dB', 'directivity_index_dB',
                 'array_gain_dB', 'processing_loss_dB'}
        field, _ = self._tl_field()
        se = sonar.passive_signal_excess_field(
            field, source_level_dB=140.0, noise_level_dB=60.0,
            directivity_index_dB=15.0, detection_threshold_dB=3.0,
        )
        assert se.sonar_budget == {
            'mode': 'passive', 'source_level_dB': 140.0,
            'detection_threshold_dB': 3.0, 'noise_level_dB': 60.0,
            'target_strength_dB': None, 'reverberation_level_dB': None,
            'directivity_index_dB': 15.0, 'array_gain_dB': None,
            'processing_loss_dB': 0.0}
        rl = np.linspace(90.0, 70.0, field.coords['range'].size)
        se_act = sonar.active_signal_excess_field(
            field, source_level_dB=220.0, target_strength_dB=10.0,
            reverberation_level_dB=rl, array_gain_dB=12.0,
            detection_threshold_dB=0.0
        )
        budget = se_act.sonar_budget
        assert set(budget) == terms
        assert budget['mode'] == 'active'
        assert budget['noise_level_dB'] is None
        assert budget['directivity_index_dB'] is None
        assert budget['reverberation_level_dB'] == rl.tolist()
        assert budget['array_gain_dB'] == 12.0

    def test_se_field_kind_and_unit(self):
        # The SE Field is tagged kind='signal_excess' and reports unit='dB'
        # — dB but neither pressure nor a loss, so .max() finds the best cell.
        field, _ = self._tl_field()
        se = sonar.passive_signal_excess_field(
            field, source_level_dB=140.0, noise_level_dB=60.0, detection_threshold_dB=0.0
        )
        assert se.kind == 'signal_excess'
        assert se.unit == 'dB'
        assert se.kind == 'signal_excess'
        se_act = sonar.active_signal_excess_field(
            field, source_level_dB=220.0, target_strength_dB=10.0,
            noise_level_dB=60.0, detection_threshold_dB=0.0
        )
        assert se_act.kind == 'signal_excess'
        assert se_act.unit == 'dB'

    def test_reverberation_length_mismatch_raises(self):
        field, _ = self._tl_field()
        with pytest.raises(
                ConfigurationError,
                match='reverberation_level_dB length .* must match'):
            sonar.active_signal_excess_field(
                field, source_level_dB=220.0, target_strength_dB=10.0,
                reverberation_level_dB=np.zeros(3), detection_threshold_dB=0.0
            )

    def test_time_series_field_rejected(self):
        from uacpy.core.results import Field
        f = Field(
            data=np.zeros(16),
            coords={'time': np.linspace(0.0, 1.0, 16)},
        )
        with pytest.raises(
                ConfigurationError,
                match='a time-domain Field is not transmission loss'):
            sonar.passive_signal_excess_field(
                f, source_level_dB=140.0, noise_level_dB=60.0, detection_threshold_dB=0.0
            )

    def test_non_field_rejected(self):
        with pytest.raises(ConfigurationError,
                           match='expected a Field, got ndarray'):
            sonar.passive_signal_excess_field(
                np.zeros((3, 3)), source_level_dB=140.0, noise_level_dB=60.0,
                detection_threshold_dB=0.0
            )

    def test_plot_signal_excess_smoke(self):
        import matplotlib.pyplot as plt
        from uacpy.visualization.plots import plot_signal_excess
        field, _ = self._tl_field()
        se = sonar.passive_signal_excess_field(
            field, source_level_dB=140.0, noise_level_dB=60.0,
            directivity_index_dB=15.0, detection_threshold_dB=20.0,
        )
        assert se.data.min() < 0.0 < se.data.max()  # boundary present
        fig, ax = plot_signal_excess(se)
        assert ax.get_xlabel() == 'Range (km)'
        # Detection boundary drawn as a contour artist beyond the heatmap.
        assert len(ax.collections) >= 2
        plt.close(fig)

    def test_plot_signal_excess_rejects_complex_and_wrong_axes(self):
        import matplotlib.pyplot as plt  # noqa: F401
        from uacpy.visualization.plots import plot_signal_excess
        field, _ = self._tl_field(complex_data=True)
        with pytest.raises(ConfigurationError,
                           match='field must carry real signal excess in dB'):
            plot_signal_excess(field)
        se = sonar.passive_signal_excess_field(
            field, source_level_dB=140.0, noise_level_dB=60.0, detection_threshold_dB=0.0
        )
        with pytest.raises(
                ConfigurationError,
                match=r"requires canonical \['depth', 'range'\] coords"):
            plot_signal_excess(se.at(depth=50.0))

    @pytest.mark.parametrize('kind', ['level', 'signal_excess',
                                      'reverberation', 'difference'])
    def test_a_field_that_is_not_tl_is_refused_as_tl(self, kind):
        """A received level (SL - TL already), the budget's own output, or a
        reverberation loss passed as TL returned a clean map (measured: a
        level field gave SE 0 where 50 dB was due). ``kind`` names the
        quantity, so each is refused by name in both budgets."""
        field, _ = self._tl_field()
        field = field.replace(kind=kind)
        with pytest.raises(ConfigurationError, match=f"a '{kind}' field"):
            sonar.passive_signal_excess_field(field, source_level_dB=140.0,
                                              noise_level_dB=60.0, detection_threshold_dB=0.0)
        with pytest.raises(ConfigurationError, match=f"a '{kind}' field"):
            sonar.active_signal_excess_field(
                field, source_level_dB=200.0, target_strength_dB=10.0,
                noise_level_dB=60.0, detection_threshold_dB=0.0)

    def test_a_loss_computed_from_pressure_is_accepted_as_tl(self):
        """Both spellings of TL pass: complex pressure and its real dB."""
        for complex_data in (False, True):
            field, tl = self._tl_field(complex_data=complex_data)
            se = sonar.passive_signal_excess_field(
                field, source_level_dB=140.0, noise_level_dB=60.0, detection_threshold_dB=0.0)
            np.testing.assert_allclose(se.data, 140.0 - 60.0 - tl,
                                       atol=1e-9)

    def test_coherent_pressure_warns_that_its_fringes_reach_the_budget(self):
        """Coherent TL overstates a detection range (measured 45 %, sonar
        guide); a complex field with a phase reference is warned about by
        name, and neither real dB nor a complex field without a phase
        reference (an incoherent sum stored complex) is."""
        import warnings as _w
        field, _ = self._tl_field(complex_data=True)
        coherent = type(field)(data=field.data, coords=dict(field.coords),
                               model='Bellhop', frequencies=2000.0,
                               phase_reference='travelling_wave')
        with pytest.warns(UserWarning, match='INCOHERENT_TL'):
            sonar.passive_signal_excess_field(coherent, source_level_dB=140.0,
                                              noise_level_dB=60.0, detection_threshold_dB=0.0)
        for quiet in (field, self._tl_field()[0]):
            with _w.catch_warnings():
                _w.simplefilter('error', UserWarning)
                sonar.passive_signal_excess_field(quiet, source_level_dB=140.0,
                                                  noise_level_dB=60.0, detection_threshold_dB=0.0)

    @pytest.mark.parametrize('run_mode', [
        'coherent_tl', 'incoherent_tl', 'semicoherent_tl', 'broadband'])
    @pytest.mark.parametrize('complex_data', [False, True])
    def test_either_the_run_mode_or_the_payload_makes_tl_coherent(
            self, run_mode, complex_data):
        """OAST returns COHERENT_TL as real dB, so the recorded run mode is
        evidence whatever the payload; and complex pressure with a phase
        reference is coherent whatever run made it — a BROADBAND result
        sliced at one frequency is the COHERENT_TL run's numbers under
        ``run_mode='broadband'``."""
        field, _ = self._tl_field(complex_data=complex_data)
        stamped = type(field)(
            data=field.data, coords=dict(field.coords), model='OAST',
            frequencies=2000.0, run_mode=run_mode,
            phase_reference='travelling_wave' if complex_data else None)
        with recorded_warnings() as rec:
            sonar.passive_signal_excess_field(
                stamped, source_level_dB=140.0, noise_level_dB=60.0,
                detection_threshold_dB=0.0)
        fired = [w for w in rec if 'INCOHERENT_TL' in str(w.message)]
        assert bool(fired) is (run_mode == 'coherent_tl' or complex_data)

    @staticmethod
    def _warns(field):
        with recorded_warnings() as rec:
            sonar.passive_signal_excess_field(
                field, source_level_dB=140.0, noise_level_dB=60.0,
                detection_threshold_dB=0.0)
        return any('INCOHERENT_TL' in str(w.message) for w in rec)

    def test_coherent_pressure_converted_to_dB_warns_about_its_fringes(self):
        field, _ = self._tl_field(complex_data=True)
        coherent = type(field)(data=field.data, coords=dict(field.coords),
                               model='Bellhop', frequencies=2000.0,
                               phase_reference='travelling_wave')
        in_dB = coherent.to_dB()
        assert not in_dB.is_complex
        assert self._warns(in_dB)

    def test_a_field_stamped_incoherent_is_silent_whatever_its_run_mode(
            self):
        # An incoherent superpose of COHERENT_TL slabs is an intensity sum
        # that inherits run_mode='coherent_tl' and stamps coherent=False.
        field, _ = self._tl_field()
        summed = type(field)(data=field.data, coords=dict(field.coords),
                             model='Kraken', frequencies=2000.0,
                             run_mode='coherent_tl',
                             coherent=False)
        assert not self._warns(summed)

    @pytest.mark.parametrize('kind', ['pressure', 'level', 'difference'])
    def test_only_signal_excess_becomes_a_detection_probability(self, kind):
        """A TL of 60 dB fed as SE read as +60 dB of excess, P_D = 1."""
        field, _ = self._tl_field()
        field = field.replace(kind=kind)
        with pytest.raises(ConfigurationError, match='not signal excess'):
            sonar.transition_probability_field(field, sigma_dB=5.6)

    def test_the_sonar_plotters_refuse_a_field_of_another_kind(self):
        """plot_signal_excess captioned a TL grid 'Signal excess'; on the
        fixed [0, 1] scale plot_detection_probability drew it as uniform
        P_D = 1. Each takes only the kind its builder tags."""
        import matplotlib.pyplot as plt
        from uacpy.visualization.plots import (plot_detection_probability,
                                               plot_signal_excess)
        field, _ = self._tl_field()
        with pytest.raises(ConfigurationError, match="'pressure' field"):
            plot_signal_excess(field)
        with pytest.raises(ConfigurationError, match="'pressure' field"):
            plot_detection_probability(field)
        se = sonar.passive_signal_excess_field(field, source_level_dB=140.0,
                                               noise_level_dB=60.0, detection_threshold_dB=0.0)
        with pytest.raises(ConfigurationError, match="'signal_excess' field"):
            plot_detection_probability(se)
        fig, _ = plot_signal_excess(se)
        plt.close(fig)
        fig, _ = plot_detection_probability(
            sonar.transition_probability_field(se, sigma_dB=5.6))
        plt.close(fig)


class TestBudgetKnobs:
    def test_array_gain_replaces_di(self):
        bg = sonar.noise_background(60.0, array_gain_dB=12.0)
        assert bg == pytest.approx(48.0)
        se = sonar.passive_signal_excess(140.0, 70.0, 60.0, array_gain_dB=12.0,
                                         detection_threshold_dB=0.0)
        assert se == pytest.approx(140.0 - 70.0 - 48.0)

    def test_array_gain_plus_di_raises(self):
        with pytest.raises(ConfigurationError,
                           match='not both — AG replaces DI'):
            sonar.noise_background(60.0, 15.0, array_gain_dB=12.0)
        with pytest.raises(ConfigurationError,
                           match='not both — AG replaces DI'):
            sonar.passive_signal_excess(
                140.0, 70.0, 60.0, directivity_index_dB=15.0, array_gain_dB=12.0,
                detection_threshold_dB=0.0
            )

    def test_array_gain_alone_uses_ag(self):
        # directivity_index_dB defaults to None ("not supplied"), so array_gain_dB
        # alone is accepted and applied — no spurious both-supplied rejection.
        assert sonar.noise_background(60.0, array_gain_dB=12.0) == pytest.approx(48.0)

    def test_di_array_with_zero_plus_ag_raises(self):
        # An explicit per-angle DI array containing a 0 is "supplied" — mixing
        # it with array_gain_dB is categorically an error (no 0.0 sentinel escape).
        with pytest.raises(ConfigurationError,
                           match='not both — AG replaces DI'):
            sonar.noise_background(60.0, np.array([0.0, 10.0]), array_gain_dB=12.0)

    def test_processing_loss_subtracts(self):
        base = sonar.passive_signal_excess(140.0, 70.0, 60.0,
                                           directivity_index_dB=15.0, detection_threshold_dB=0.0)
        lossy = sonar.passive_signal_excess(140.0, 70.0, 60.0,
                                            directivity_index_dB=15.0,
                                            processing_loss_dB=3.0, detection_threshold_dB=0.0)
        assert lossy == pytest.approx(base - 3.0)
        fom = sonar.figure_of_merit(140.0, 60.0, 15.0,
                                    processing_loss_dB=3.0, detection_threshold_dB=0.0)
        assert fom == pytest.approx(140.0 - 45.0 - 3.0)

    def test_active_array_gain_applies_to_noise_not_reverb(self):
        # Reverb-only: AG must change nothing.
        se_rl = sonar.active_signal_excess(
            220.0, 60.0, 10.0, reverberation_level_dB=80.0, detection_threshold_dB=0.0
        )
        se_rl_ag = sonar.active_signal_excess(
            220.0, 60.0, 10.0, reverberation_level_dB=80.0, array_gain_dB=12.0,
            detection_threshold_dB=0.0
        )
        assert se_rl_ag == pytest.approx(se_rl)
        # Noise-only: AG acts exactly like DI of the same value.
        se_di = sonar.active_signal_excess(
            220.0, 60.0, 10.0, noise_level_dB=70.0, directivity_index_dB=12.0,
            detection_threshold_dB=0.0
        )
        se_ag = sonar.active_signal_excess(
            220.0, 60.0, 10.0, noise_level_dB=70.0, array_gain_dB=12.0, detection_threshold_dB=0.0
        )
        assert se_ag == pytest.approx(se_di)

    def test_field_variants_thread_knobs(self):
        field, tl = TestSignalExcessField._tl_field()
        se = sonar.passive_signal_excess_field(
            field, source_level_dB=140.0, noise_level_dB=60.0,
            array_gain_dB=12.0, processing_loss_dB=3.0, detection_threshold_dB=0.0
        )
        expected = 140.0 - tl - 48.0 - 3.0
        np.testing.assert_allclose(se.data, expected)
        assert se.sonar_budget['array_gain_dB'] == 12.0
        assert se.sonar_budget['processing_loss_dB'] == 3.0


class TestScanFalseAlarmAccountsForManyLooks:
    """Keeping the largest of many looks is many chances to false-alarm.

    A scanning sonar forms every beam and reports the biggest. Judged with a
    single-beam P_F its false-alarm rate is that of the whole scan, which is
    larger; these convert between the two so a threshold can be set for the
    detector actually used.
    """

    def test_one_look_is_the_identity(self):
        assert sonar.per_look_false_alarm(1e-4, 1) == pytest.approx(1e-4)
        assert sonar.scan_false_alarm(1e-4, 1) == pytest.approx(1e-4)

    def test_the_two_are_inverses(self):
        for pf, n in ((1e-4, 16), (1e-2, 3), (1e-6, 100)):
            per = sonar.per_look_false_alarm(pf, n)
            assert sonar.scan_false_alarm(per, n) == pytest.approx(pf)

    def test_many_looks_need_a_tighter_per_look_rate(self):
        per = sonar.per_look_false_alarm(1e-4, 16)
        assert per < 1e-4
        # Small rates are very nearly pf/n, but not exactly - the exact form
        # is what keeps the round trip above exact.
        assert per == pytest.approx(1e-4 / 16, rel=1e-3)
        assert per != 1e-4 / 16

    def test_a_scan_raises_the_rate_it_is_given(self):
        assert sonar.scan_false_alarm(1e-4, 16) > 1e-4
        assert sonar.scan_false_alarm(1e-4, 16) == pytest.approx(
            1.0 - (1.0 - 1e-4) ** 16)

    def test_it_raises_the_detection_threshold(self):
        kw = dict(pd=0.5, bandwidth_hz=10.0, integration_time_s=10.0)
        one = sonar.detection_threshold_energy(pf=1e-4, **kw)
        scan = sonar.detection_threshold_energy(
            pf=sonar.per_look_false_alarm(1e-4, 16), **kw)
        assert scan > one

    def test_a_fractional_look_count_is_ACCEPTED(self):
        """Resolution cells do not come out whole, and the caller needs that.

        ``independent_beams`` returns 16.97 for the array in example 42, and
        rounding it would be a silent change of threshold.
        """
        assert sonar.per_look_false_alarm(1e-4, 16.97) == pytest.approx(
            1.0 - (1.0 - 1e-4) ** (1.0 / 16.97))

    def test_the_guard_sits_at_one_look_not_at_zero(self):
        """Both sides of the boundary: fewer than one look is meaningless."""
        assert sonar.per_look_false_alarm(1e-4, 1.0) == pytest.approx(1e-4)
        for bad in (0.999, 0.0, -3.0):
            with pytest.raises(ConfigurationError, match='n_looks'):
                sonar.per_look_false_alarm(1e-4, bad)
            with pytest.raises(ConfigurationError, match='n_looks'):
                sonar.scan_false_alarm(1e-4, bad)

    def test_it_survives_a_tiny_rate_over_many_looks(self):
        """``1 - (1-pf)**(1/n)`` underflows to exactly 0 and loses the inverse.

        At pf=1e-12 over 1e5 looks the naive form returns 0.0, which then
        makes scan_false_alarm raise on its own output.
        """
        per = sonar.per_look_false_alarm(1e-12, 1e5)
        assert per > 0.0
        assert sonar.scan_false_alarm(per, 1e5) == pytest.approx(1e-12,
                                                                rel=1e-6)

    def test_a_non_finite_look_count_is_refused(self):
        for bad in (np.nan, np.inf):
            with pytest.raises(ConfigurationError, match='n_looks'):
                sonar.per_look_false_alarm(1e-4, bad)


class TestFieldArrayGainMayVaryPerSample:
    """AG is a per-sample quantity whenever the signal is not a plane wave.

    A beam's realised gain changes with target depth and range, because a
    multi-mode arrival puts a different share of its energy inside the main
    lobe at every point. The scalar form stays the common case; these pin
    that a grid of AG is accepted and applied sample by sample.
    """

    def test_grid_of_gains_applies_sample_by_sample(self):
        field, tl = TestSignalExcessField._tl_field()
        ag = np.linspace(6.0, 14.0, tl.size).reshape(tl.shape)
        se = sonar.passive_signal_excess_field(
            field, source_level_dB=140.0, noise_level_dB=60.0, array_gain_dB=ag,
            detection_threshold_dB=0.0
        )
        np.testing.assert_allclose(se.data, 140.0 - tl - (60.0 - ag))

    def test_grid_of_gains_is_recorded_in_full_and_summarised(self):
        field, tl = TestSignalExcessField._tl_field()
        # Deliberately SKEWED: a linspace has mean == median exactly, so a
        # summary that quietly reported the mean would pass unnoticed.
        ag = np.full(tl.size, 6.0)
        ag[-1] = 14.0
        ag = ag.reshape(tl.shape)
        assert not np.isclose(np.mean(ag), np.median(ag))
        se = sonar.passive_signal_excess_field(
            field, source_level_dB=140.0, noise_level_dB=60.0, array_gain_dB=ag,
            detection_threshold_dB=0.0
        )
        recorded = np.asarray(se.sonar_budget['array_gain_dB'])
        assert recorded.shape == ag.shape
        assert np.array_equal(recorded, ag)
        summary = sonar.SonarBudget.from_dict(
            se.sonar_budget).summary()
        assert (f"AG 6-14 dB (median {float(np.median(ag)):g} dB) over "
                f"{ag.size} samples") in summary

    def test_a_scalar_gain_stays_a_scalar_in_the_budget(self):
        field, _ = TestSignalExcessField._tl_field()
        se = sonar.passive_signal_excess_field(
            field, source_level_dB=140.0, noise_level_dB=60.0, array_gain_dB=12.0,
            detection_threshold_dB=0.0
        )
        assert se.sonar_budget['array_gain_dB'] == 12.0

    def test_a_gain_that_does_not_fit_the_grid_is_refused(self):
        field, tl = TestSignalExcessField._tl_field()
        bad = np.linspace(6.0, 14.0, tl.shape[0] + 1)
        with pytest.raises(ConfigurationError, match='array_gain_dB'):
            sonar.passive_signal_excess_field(
                field, source_level_dB=140.0, noise_level_dB=60.0, array_gain_dB=bad,
                detection_threshold_dB=0.0
            )

    def test_the_active_twin_takes_a_grid_too(self):
        field, tl = TestSignalExcessField._tl_field()
        ag = np.full(tl.shape, 9.0)
        se_grid = sonar.active_signal_excess_field(
            field, source_level_dB=220.0, target_strength_dB=10.0,
            noise_level_dB=70.0, array_gain_dB=ag, detection_threshold_dB=0.0
        )
        se_scalar = sonar.active_signal_excess_field(
            field, source_level_dB=220.0, target_strength_dB=10.0,
            noise_level_dB=70.0, array_gain_dB=9.0, detection_threshold_dB=0.0
        )
        np.testing.assert_allclose(se_grid.data, se_scalar.data)


class TestDetectionProbabilityField:
    @staticmethod
    def _se_field(values):
        from uacpy.core.results import Field
        values = np.asarray(values, dtype=float)
        return Field(
            data=values,
            coords={'depth': np.arange(values.shape[0], dtype=float),
                    'range': np.arange(values.shape[1], dtype=float) + 1.0},
            model='Bellhop', kind='signal_excess',
        )

    def test_transition_curve_anchors(self):
        # Pd = Phi(SE/sigma): 0.5 at SE=0, Phi(±1) at SE=±sigma.
        sigma = 5.6
        se = self._se_field([[0.0, sigma, -sigma, 2 * sigma]])
        pd = sonar.transition_probability_field(se, sigma_dB=sigma)
        expected = norm.cdf(np.array([0.0, 1.0, -1.0, 2.0]))
        np.testing.assert_allclose(pd.data[0], expected, atol=1e-12)
        assert pd.sigma_dB == pytest.approx(sigma)
        assert list(pd.coords) == ['depth', 'range']

    def test_pd_field_kind_and_unit(self):
        # The P_D Field is tagged kind='probability_of_detection' with
        # unit='1' — a dimensionless 0-1 probability, not dB.
        se = self._se_field([[0.0, 6.0]])
        pd = sonar.transition_probability_field(se, sigma_dB=6.0)
        assert pd.kind == 'probability_of_detection'
        assert pd.unit == '1'
        assert pd.kind == 'probability_of_detection'
        assert pd.unit == '1'

    def test_monotonic_in_se_and_bounded(self):
        se = self._se_field(np.linspace(-30, 30, 61).reshape(1, -1))
        pd = sonar.transition_probability_field(se, sigma_dB=6.0)
        assert np.all(np.diff(pd.data[0]) > 0)
        assert pd.data.min() >= 0.0 and pd.data.max() <= 1.0

    def test_bad_sigma_raises(self):
        se = self._se_field([[0.0]])
        with pytest.raises(ConfigurationError,
                           match='sigma_dB must be > 0 dB and finite'):
            sonar.transition_probability_field(se, sigma_dB=0.0)
        with pytest.raises(ConfigurationError,
                           match='sigma_dB must be > 0 dB and finite'):
            sonar.transition_probability_field(se, sigma_dB=-1.0)

    @pytest.mark.parametrize('bad', [np.nan, np.inf])
    def test_a_non_finite_sigma_raises_rather_than_filling_nan(self, bad):
        se = self._se_field([[0.0]])
        with pytest.raises(ConfigurationError, match='sigma_dB must be'):
            sonar.transition_probability_field(se, sigma_dB=bad)

    def test_non_field_and_complex_rejected(self):
        with pytest.raises(ConfigurationError,
                           match='expected a Field, got ndarray'):
            sonar.transition_probability_field(
                np.zeros((2, 2)), sigma_dB=6.0,
            )
        from uacpy.core.results import Field
        cplx = Field(
            data=np.zeros((1, 2), dtype=complex),
            coords={'depth': [0.0], 'range': [1.0, 2.0]},
            kind='signal_excess',
        )
        with pytest.raises(ConfigurationError,
                           match='field must carry real signal excess in dB'):
            sonar.transition_probability_field(cplx, sigma_dB=6.0)

    def test_detection_range_by_depth(self):
        # Row 0 crosses zero between samples; row 1 all positive (inf);
        # row 2 all negative (nan).
        r = np.array([1000.0, 2000.0, 3000.0])
        se = self._se_field([[10.0, 0.0, -10.0],
                             [5.0, 5.0, 5.0],
                             [-5.0, -5.0, -5.0]])
        se.coords['range'] = r
        depths, dr = sonar.detection_ranges_by_depth(se)
        assert depths.shape == dr.shape == (3,)
        assert dr[0] == pytest.approx(2000.0)   # exact zero at 2 km
        assert np.isinf(dr[1])
        assert np.isnan(dr[2])

    @pytest.mark.parametrize('kind', ['pressure', 'level', 'difference'])
    def test_detection_ranges_by_depth_refuses_a_field_that_is_not_signal_excess(
            self, kind):
        """A TL of 47-80 dB read as SE is positive everywhere: inf range at
        every depth."""
        se = self._se_field([[60.0, 70.0, 80.0]])
        with pytest.raises(ConfigurationError, match='not signal excess'):
            sonar.detection_ranges_by_depth(se.replace(kind=kind))

    def test_detection_ranges_by_depth_accepts_a_signal_excess_field(self):
        se = self._se_field([[60.0, 70.0, 80.0]])
        _, dr = sonar.detection_ranges_by_depth(se)
        assert np.isinf(dr[0])

    def test_detection_range_by_depth_requires_canonical(self):
        from uacpy.core.results import Field
        f = Field(data=np.zeros(4), coords={'range': np.arange(4.0) + 1},
                  kind='signal_excess')
        with pytest.raises(
                ConfigurationError,
                match=r"requires canonical \['depth', 'range'\] coords"):
            sonar.detection_ranges_by_depth(f)

    def test_plot_detection_probability_smoke(self):
        import matplotlib.pyplot as plt
        from uacpy.visualization.plots import plot_detection_probability
        se = self._se_field(
            np.linspace(20, -20, 40).reshape(2, 20),
        )
        pd = sonar.transition_probability_field(se, sigma_dB=5.6)
        fig, ax = plot_detection_probability(pd)
        assert ax.get_xlabel() == 'Range (km)'
        assert 'σ = 5.6 dB' in ax.get_title()
        plt.close(fig)
        with pytest.raises(
                ConfigurationError,
                match=r"requires canonical \['depth', 'range'\] coords"):
            plot_detection_probability(pd.at(depth=0.0))


class TestTargetStrength:
    def test_two_metre_sphere_is_zero_dB(self):
        # The classic anchor (Urick Table 9.1; Abraham §3.4 uses it too):
        # a = 2 m -> a²/4 = 1 m² -> TS = 0 dB.
        assert sonar.ts_sphere(2.0) == pytest.approx(0.0)

    def test_sphere_frequency_flat_and_scales(self):
        # TS = 10log10(a²/4): doubling the radius adds 6.02 dB.
        assert (sonar.ts_sphere(4.0) - sonar.ts_sphere(2.0)
                == pytest.approx(20.0 * np.log10(2.0)))

    def test_convex_reduces_to_sphere(self):
        assert sonar.ts_convex(2.0, 2.0) == pytest.approx(sonar.ts_sphere(2.0))

    def test_ellipsoid_consistency_chain(self):
        # b = c = a recovers the sphere (principal radii b²/a = a).
        assert sonar.ts_ellipsoid(2.0, 2.0, 2.0) == pytest.approx(
            sonar.ts_sphere(2.0))
        # Urick Table 9.1 form: TS = 20log10(bc/2a) along axis a.
        a, b, c = 10.0, 2.0, 1.5
        assert sonar.ts_ellipsoid(a, b, c) == pytest.approx(
            20.0 * np.log10(b * c / (2.0 * a)))

    def test_cylinder_broadside_anchor(self):
        # Abraham Fig. 3.24 setup: a=1 m, L=5 m, fc=1 kHz (λ=1.5 m):
        # TS = 10log10(aL²/2λ) = 10log10(8.333) = 9.21 dB.
        ts = sonar.ts_cylinder(1.0, 5.0, frequency=1000.0, sound_speed=1500.0)
        assert ts == pytest.approx(9.208, abs=0.01)

    def test_cylinder_aspect_pattern(self):
        # First null at β = kL·sinθ = π, i.e. sinθ = λ/(2L)
        # (Abraham §3.4: null-to-null main lobe ≈ λ/L).
        L, f, c = 5.0, 1000.0, 1500.0
        lam = c / f
        theta_null = np.degrees(np.arcsin(lam / (2.0 * L)))
        ts_null = sonar.ts_cylinder(1.0, L, frequency=f, angle_deg=theta_null,
                                    sound_speed=c)
        # Evaluated exactly on the analytic null, sinc^2 bottoms out at the
        # float64 sin() floor: TS is -306 dB here. -60 dB is a "this is a
        # null, not a lobe" threshold with ~245 dB of headroom, so it does not
        # pin the null depth, only its location.
        assert ts_null < -60.0
        # Monotone decrease across the main lobe.
        angles = np.linspace(0.0, theta_null * 0.95, 10)
        ts = sonar.ts_cylinder(1.0, L, frequency=f, angle_deg=angles, sound_speed=c)
        assert np.all(np.diff(ts) < 0)

    def test_plate_normal_incidence(self):
        # TS = 20log10(ab/λ): 1 m × 1 m plate at λ = 1 m -> 0 dB.
        ts = sonar.ts_plate(1.0, 1.0, frequency=1500.0, sound_speed=1500.0)
        assert ts == pytest.approx(0.0)

    def test_geometric_regime_warning(self):
        # ka = 2π·50/1500·0.5 ≈ 0.105 « 10 -> Rayleigh regime, must warn.
        with pytest.warns(UserWarning, match="geometric"):
            sonar.ts_sphere(0.5, frequency=50.0)
        # No frequency -> no check, no warning.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            sonar.ts_sphere(0.5)

    def test_cylinder_and_plate_warn_below_ka_one(self):
        # The cylinder formula holds for ka > 1 on the RADIUS (Urick's
        # ka >> 1; Abraham's reference cylinder runs at ka ~ 4).
        # a = 0.05 m at 1 kHz -> ka = 2*pi*1000/1500*0.05 = 0.209 < 1: warn.
        with pytest.warns(UserWarning, match="geometric"):
            sonar.ts_cylinder(0.05, 5.0, frequency=1000.0)
        # The plate check scale is a full dimension, so its bound is one
        # wavelength (kw > 2*pi), not ka > 1: min(0.1, 0.2) = 0.1 m at
        # lambda = 1.5 m warns, and so does a 1 m plate (0.67 lambda).
        with pytest.warns(UserWarning, match="geometric"):
            sonar.ts_plate(0.1, 0.2, frequency=1000.0)
        with pytest.warns(UserWarning, match="geometric"):
            sonar.ts_plate(1.0, 1.0, frequency=1000.0)
        # Above the bounds both stay silent: cylinder a = 0.5 m -> ka = 2.09;
        # a 3 m plate is 2 wavelengths across at 1 kHz.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            sonar.ts_cylinder(0.5, 5.0, frequency=1000.0)
            sonar.ts_plate(3.0, 3.0, frequency=1000.0)

    def test_plate_warns_in_rayleigh_regime(self):
        # A 0.2 x 0.2 m plate at lambda = 1 m is 0.2 wavelengths across —
        # deep in the Rayleigh regime, where physical optics does not apply.
        # It passes the ka > 1 bound, so only the wavelength check flags it;
        # the level itself is the plate's -28.0 dB either way, which is what
        # makes the warning the whole of the protection here.
        with pytest.warns(UserWarning, match="geometric"):
            ts = sonar.ts_plate(0.2, 0.2, frequency=1500.0, sound_speed=1500.0)
        assert ts == pytest.approx(20.0 * np.log10(0.04))

    @pytest.mark.parametrize('call', [
        lambda f, **kw: sonar.ts_cylinder(1.0, 10.0, frequency=f, **kw),
        lambda f, **kw: sonar.ts_plate(2.0, 3.0, frequency=f, **kw),
    ])
    def test_a_frequency_array_is_the_curve_of_single_calls(self, call):
        f = np.array([2000.0, 5000.0, 12000.0])
        np.testing.assert_array_equal(call(f), [call(fi) for fi in f])
        assert isinstance(call(5000.0), float)
        grid = call(f[:, None], angle_deg=np.array([0.0, 3.0]))
        assert grid.shape == (3, 2)
        np.testing.assert_array_equal(
            grid[:, 1], [call(fi, angle_deg=3.0) for fi in f])
        with pytest.raises(ConfigurationError, match='frequency'):
            call(np.array([2000.0, -1.0]))

    def test_the_frequency_array_check_reads_the_lowest_frequency(self):
        # ka = 2*pi*f*a/c: 100 Hz on a 1 m cylinder is 0.42, 2 kHz is 8.4.
        with pytest.warns(UserWarning, match=r"k·a = 0\.42"):
            sonar.ts_cylinder(1.0, 10.0, frequency=np.array([2000.0, 100.0]))

    def test_needle_ellipsoid_warns_although_k_times_semi_axis_is_geometric(self):
        """``ts_ellipsoid`` delegates to ``ts_convex(b²/a, c²/a)``, so its
        ``ka > 10`` check runs on the principal radii of curvature at the tip
        of the ``a`` axis — not on any semi-axis. A needle ``a = 10 m``,
        ``b = c = 1 m`` at 1 kHz has ``k·a = 41.9``, comfortably geometric,
        while the radii are ``b²/a = 0.1 m`` and the check sees 0.42. The
        message carries the number so a reader who did the semi-axis
        arithmetic can see which radius was tested, and it names
        ``ts_ellipsoid``, the function that was called.
        """
        with pytest.warns(UserWarning, match="geometric") as caught:
            sonar.ts_ellipsoid(10.0, 1.0, 1.0, frequency=1000.0,
                               sound_speed=1500.0)
        message = str(caught[0].message)
        assert 'k·a = 0.42' in message
        assert '41.9' not in message
        assert message.startswith('ts_ellipsoid:')

    def test_oblate_ellipsoid_is_silent_although_k_times_semi_axis_is_not(self):
        """The same substitution run the other way: an oblate ``a = 1 m``,
        ``b = c = 10 m`` at 1 kHz has ``k·a = 4.19``, below the bound, while
        the radii are ``b²/a = 100 m`` and the check sees 419. Silence here
        and a warning in the needle case together say WHICH radius is
        tested; either one alone is also explained by a semi-axis check.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sonar.ts_ellipsoid(1.0, 10.0, 10.0, frequency=1000.0,
                               sound_speed=1500.0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ConfigurationError, match='radius_m must be > 0'):
            sonar.ts_sphere(0.0)
        with pytest.raises(ConfigurationError,
                           match='frequency must be > 0'):
            sonar.ts_cylinder(1.0, 5.0, frequency=-100.0)
        with pytest.raises(ConfigurationError, match='height_m must be > 0'):
            sonar.ts_plate(1.0, -1.0, frequency=1000.0)

    def test_feeds_active_signal_excess(self):
        ts = sonar.ts_cylinder(0.5, 10.0, frequency=2000.0)
        se = sonar.active_signal_excess(
            190.0, 60.0, ts, noise_level_dB=75.0, directivity_index_dB=15.0,
            detection_threshold_dB=0.0
        )
        assert np.isfinite(se)


class TestDetectionThresholdReference:
    """Pin the DT convention and the 10*log10(w) offset to Urick's form."""

    def test_five_dB_per_decade_of_time_bandwidth(self):
        """Abraham 9.2.3.1: SNR_d falls 5 dB per decade of M = w*t."""
        from uacpy.sonar import detection_threshold_energy
        a = detection_threshold_energy(0.5, 1e-4, bandwidth_hz=100.0,
                                       integration_time_s=1.0, exact=False)
        b = detection_threshold_energy(0.5, 1e-4, bandwidth_hz=100.0,
                                       integration_time_s=10.0, exact=False)
        assert (a - b) == pytest.approx(5.0, abs=1e-9)

    def test_offset_from_urick_band_power_form_is_10log10_w(self):
        """This DT is the unitless S0/N0 ratio; Urick's d*w/t form is
        referenced to noise in a 1-Hz band. They differ by 10*log10(w)."""
        import numpy as np
        from uacpy.sonar import detection_threshold_energy, detection_index
        pd, pf, w, t = 0.5, 1e-4, 100.0, 2.0
        this = detection_threshold_energy(pd, pf, w, t, exact=False)
        urick = 5.0 * np.log10(detection_index(pd, pf) * w / t)
        assert (urick - this) == pytest.approx(10.0 * np.log10(w), abs=1e-9)
        # 100 Hz -> exactly the 20 dB the docstring warns about
        assert (urick - this) == pytest.approx(20.0, abs=1e-9)


class TestDetectionThresholdLargeMEnvelope:
    """``DT = 5*log10(d/M)`` is Abraham eq. (2.77), the large-``M`` limit of
    (2.76), and it is optimistic at every operating point.

    The exact noise-normalised energy detector has ``T ~ Gamma(M, 1)`` under
    ``H0`` and ``(1+S)*Gamma(M, 1)`` under ``H1``, so the required per-cell
    SNR is ``S = Ginv(1-Pf; M)/Ginv(1-Pd; M) - 1``. Feeding that ``S`` back
    through the Gamma survival function recovers the requested operating
    point, which is what makes it the benchmark.
    """

    @staticmethod
    def _exact_dt_dB(pd, pf, m):
        from scipy.stats import gamma
        return 10.0 * np.log10(gamma.isf(pf, m) / gamma.isf(pd, m) - 1.0)

    def test_the_exact_benchmark_recovers_its_own_operating_point(self):
        from scipy.stats import gamma
        pd, pf, m = 0.9, 1e-6, 40.0
        s = 10.0 ** (self._exact_dt_dB(pd, pf, m) / 10.0)
        h = gamma.isf(pf, m)
        assert gamma.sf(h, m) == pytest.approx(pf, rel=1e-9)
        assert gamma.sf(h / (1.0 + s), m) == pytest.approx(pd, rel=1e-9)

    @pytest.mark.parametrize('pd, pf, m, expected_dB', [
        (0.9, 1e-6, 1.0, -13.337),
        (0.9, 1e-6, 10.0, -3.485),
        (0.9, 1e-6, 100.0, -1.070),
        (0.99, 1e-6, 1.0, -22.879),
    ])
    def test_the_shipped_value_is_optimistic_by_the_documented_amount(
            self, pd, pf, m, expected_dB):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got = detection_threshold_energy(pd, pf, m, 1.0, exact=False)
        error = got - self._exact_dt_dB(pd, pf, m)
        assert error < 0.0
        assert error == pytest.approx(expected_dB, abs=5e-3)

    def test_the_warning_boundary_is_the_1_dB_promise_itself(self):
        """The guard's threshold and the accuracy claim are one thing, so the
        two sides of it are the two sides of the promise, read at the 0.01 dB
        the warning prints. At pd=0.9, pf=1e-6 the error is 1.006 dB at
        M = 113 and 1.002 dB at M = 114, which prints as 1.00."""
        pd, pf = 0.9, 1e-6
        # The measurements run under an explicit filter so the boundary is
        # pinned to the code, not to whatever -W the suite is invoked with.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert round(abs(self._exact_dt_dB(pd, pf, 113.0)
                             - detection_threshold_energy(pd, pf, 113.0, 1.0, exact=False)),
                         2) > 1.0
            assert round(abs(self._exact_dt_dB(pd, pf, 114.0)
                             - detection_threshold_energy(pd, pf, 114.0, 1.0, exact=False)),
                         2) <= 1.0
        with pytest.warns(UserWarning, match='optimistic by 1.01 dB'):
            detection_threshold_energy(pd, pf, 113.0, 1.0, exact=False)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            detection_threshold_energy(pd, pf, 114.0, 1.0, exact=False)

    def test_the_warning_names_the_measured_error_at_this_operating_point(self):
        with pytest.warns(UserWarning) as record:
            detection_threshold_energy(0.9, 1e-6, 10.0, 1.0, exact=False)
        message = str(record[0].message)
        assert 'optimistic by 3.49 dB' in message
        assert 'exact threshold 6.29 dB' in message

    @pytest.mark.parametrize('label, pd, pf, m', [
        ('guide DT_PASSIVE', 0.5, 1e-4, 500.0),
        ('guide DT_ACTIVE', 0.5, 1e-4, 50.0),
        ('test_detection_threshold_energy_formula', 0.9, 0.01, 100.0),
        ('test_five_dB_per_decade', 0.5, 1e-4, 100.0),
        ('test_offset_from_urick', 0.5, 1e-4, 200.0),
        ('example_27', 0.5, 1e-4, 100.0),
    ])
    def test_no_documented_operating_point_warns(self, label, pd, pf, m):
        """A check that fires on the package's own worked examples trains
        users to ignore it. Every anchor the guide, the suite and the examples
        publish is inside the 1 dB promise, so every one must stay silent — and
        the assertion is on the measured error, not on the guard, so it also
        catches the envelope drifting off the promise."""
        error = (detection_threshold_energy(pd, pf, m, 1.0, exact=False)
                 - self._exact_dt_dB(pd, pf, m))
        assert abs(error) < 1.0, f'{label} is outside the promise: {error} dB'
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            detection_threshold_energy(pd, pf, m, 1.0, exact=False)

    def test_the_fallback_bound_covers_an_unresolvable_operating_point(self):
        """The exact benchmark is allowed to fail; the check is not allowed to
        disappear when it does."""
        from uacpy.sonar.detection import _exact_detection_threshold_dB
        # A sub-unity time-bandwidth product at a low Pf: the detection index
        # is a healthy 36, so this is not the degenerate pd == pf corner, but
        # the Gamma quantile ratio overflows and the exact value is not finite.
        pd, pf, m = 0.5, 1e-9, 1e-6
        assert sonar.detection_index(pd, pf) > 0
        assert not np.isfinite(_exact_detection_threshold_dB(pd, pf, m))
        with pytest.warns(UserWarning, match='fitted fallback bound'):
            detection_threshold_energy(pd, pf, m, 1.0, exact=False)

    @pytest.mark.parametrize('pd, pf, m', [
        (0.9, 1e-6, 1.0), (0.9, 1e-6, 10.0), (0.5, 1e-4, 500.0)])
    def test_the_exact_threshold_the_warning_names_is_public(self, pd, pf, m):
        """The warning's remedy was a private function. ``exact=True`` returns
        the benchmark itself, silently, on both sides of the 1 dB promise, and
        the warning names that keyword."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            got = detection_threshold_energy(pd, pf, m, 1.0, exact=True)
        assert got == pytest.approx(self._exact_dt_dB(pd, pf, m), abs=1e-12)
        if m < 114.0:
            with pytest.warns(UserWarning, match='exact=True'):
                detection_threshold_energy(pd, pf, m, 1.0, exact=False)

    def test_an_unresolvable_exact_threshold_is_nan_not_the_approximation(self):
        pd, pf, m = 0.5, 1e-9, 1e-6
        with pytest.warns(UserWarning, match='did not resolve'):
            got = detection_threshold_energy(pd, pf, m, 1.0, exact=True)
        assert np.isnan(got)

    def test_the_warning_does_not_change_the_returned_value(self):
        pd, pf, m = 0.9, 1e-6, 10.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got = detection_threshold_energy(pd, pf, m, 1.0, exact=False)
        assert got == pytest.approx(
            5.0 * np.log10(sonar.detection_index(pd, pf) / m), abs=1e-12)


class TestTargetStrengthAgainstAbraham:
    """Abraham §3.4: eq. (3.217) ``a1*a2/4`` for a smooth convex body,
    eq. (3.218) ``a**2/4`` for a rigid sphere (geometric regime, ka > 10),
    eq. (3.221) ``(a*L**2/2*lam)*[sin(b)/b]**2*cos(th)**2`` for a cylinder with
    ``b = k*L*sin(th)``, and the physical-optics ``A**2/lam**2`` for a plate."""

    C, A, L, F = 1500.0, 0.5, 10.0, 5000.0

    def test_sphere_and_its_degenerate_cases(self):
        from uacpy.sonar.target_strength import (ts_convex, ts_ellipsoid,
                                                 ts_sphere)
        assert ts_sphere(2.0) == pytest.approx(0.0)      # Urick's TS=0 anchor
        for a in (0.7, 2.5, 9.0):
            assert ts_sphere(a) == pytest.approx(10 * np.log10(a ** 2 / 4))
            assert ts_convex(a, a) == pytest.approx(ts_sphere(a))
            assert ts_ellipsoid(a, a, a) == pytest.approx(ts_sphere(a))

    def test_cylinder_matches_equation_3_221_at_every_aspect(self):
        from uacpy.sonar.target_strength import ts_cylinder
        lam = self.C / self.F
        k = 2 * np.pi / lam
        for ang in (0.0, 0.25, 0.5, 1.0, 2.0):
            th = np.deg2rad(ang)
            beta = k * self.L * np.sin(th)
            # np.sinc is normalised, sinc(x) = sin(pi x)/(pi x), so beta/pi
            # recovers Abraham's unnormalised [sin b / b]. This mirrors the
            # implementation's own idiom, so the beta -> 0 limit at broadside
            # is asserted against the same construction rather than
            # independently.
            ref = 10 * np.log10(self.A * self.L ** 2 / (2 * lam)
                                * np.sinc(beta / np.pi) ** 2 * np.cos(th) ** 2)
            assert ts_cylinder(self.A, self.L, frequency=self.F,
                               angle_deg=ang) == pytest.approx(ref)

    def test_cylinder_main_lobe_is_the_documented_width(self):
        """The docstring promises a null-to-null width of ~lam/L radians, i.e.
        the first null where ``beta = pi`` (``sin th = lam/2L``)."""
        from uacpy.sonar.target_strength import ts_cylinder
        lam = self.C / self.F
        first_null = np.degrees(np.arcsin(lam / (2 * self.L)))
        # -309 dB at the analytic null (float64 sin() floor); -100 dB only
        # asserts that a null lands here, not how deep it is.
        assert ts_cylinder(self.A, self.L, frequency=self.F,
                           angle_deg=first_null) < -100.0
        assert 2 * first_null == pytest.approx(np.degrees(lam / self.L),
                                               rel=1e-3)

    def test_plate_at_normal_incidence_is_physical_optics(self):
        from uacpy.sonar.target_strength import ts_plate
        lam = self.C / self.F
        w, h = 4.0, 3.0
        assert ts_plate(w, h, frequency=self.F) == pytest.approx(20 * np.log10(w * h / lam))


class TestReverberationCells:
    """The scattering cell is ``R*phi*(c*tau/2)`` for a boundary and
    ``R*phi*(c*tau/2)*R*theta`` for a volume (Stergiopoulos, *Advanced Signal
    Processing Handbook*; Urick Ch. 8), so with spherical spreading boundary
    reverberation falls 30 dB per decade of range and volume 20 dB."""

    KW = dict(pulse_length_s=0.01, sound_speed=1500.0)
    SL, SB, SV, PHI, PSI = 200.0, -27.0, -70.0, 0.1, 0.01

    def test_cells_match_the_published_expressions(self):
        from uacpy.sonar.reverberation import (boundary_reverberation,
                                               volume_reverberation)
        r = np.array([500.0, 2000.0, 9000.0])
        tl = 20 * np.log10(r)
        c, tau = self.KW['sound_speed'], self.KW['pulse_length_s']
        assert np.allclose(
            boundary_reverberation(r, self.SL, self.SB,
                                   horizontal_beamwidth_rad=self.PHI, **self.KW),
            self.SL - 2 * tl + self.SB + 10 * np.log10(self.PHI * r * (c * tau / 2)))
        assert np.allclose(
            volume_reverberation(r, self.SL, self.SV,
                                 solid_angle_beamwidth_sr=self.PSI, **self.KW),
            self.SL - 2 * tl + self.SV + 10 * np.log10(self.PSI * r ** 2 * (c * tau / 2)))

    def test_decay_slopes_are_the_classic_minus_30_and_minus_20(self):
        from uacpy.sonar.reverberation import (boundary_reverberation,
                                               volume_reverberation)
        r = np.array([1e3, 1e4])                       # one decade
        b = boundary_reverberation(r, self.SL, self.SB,
                                   horizontal_beamwidth_rad=self.PHI, **self.KW)
        v = volume_reverberation(r, self.SL, self.SV,
                                 solid_angle_beamwidth_sr=self.PSI, **self.KW)
        assert b[1] - b[0] == pytest.approx(-30.0)
        assert v[1] - v[0] == pytest.approx(-20.0)

    def test_total_is_an_incoherent_power_sum(self):
        from uacpy.sonar.reverberation import total_reverberation
        got = total_reverberation(np.array([80.0]), np.array([74.0]))
        assert got[0] == pytest.approx(10 * np.log10(10 ** 8.0 + 10 ** 7.4))


class TestReverberationGuardsItsDomain:
    """The cell-scattering form takes the scattering area as
    ``Phi * r * (c*tau/2)`` — an annulus of width ``c*tau/2`` treated as if it
    all sat at range ``r``. The exact annulus between ``r`` and ``r + c*tau/2``
    has area ``Phi * (r2**2 - r1**2) / 2``, so the approximation is low by
    exactly ``10*log10(1 + c*tau/(4*r))``: 0.022 dB at ``c*tau/2 = r/100``,
    0.212 dB at ``r/10``, 0.969 dB at ``r/2``, 1.761 dB once the cell is as
    long as the range to it.

    Nothing said so. A 75 m cell at r = 1 m returned a confident 178.75 dB, a
    negative range returned NaN carrying only numpy's ``invalid value
    encountered in log10``, and r = 0 fell out of an inf - inf.
    """

    KW = dict(pulse_length_s=0.1, horizontal_beamwidth_rad=0.1,
              sound_speed=1500.0)

    def test_the_derived_error_matches_the_exact_annulus_area(self):
        # Pins the expression the warning quotes, independently of the module.
        c, tau, phi = 1500.0, 0.1, 0.1
        extent = c * tau / 2.0
        for r in (10.0, 75.0, 150.0, 750.0):
            exact = phi * ((r + extent) ** 2 - r ** 2) / 2.0
            approx = phi * r * extent
            predicted = 10 * np.log10(1.0 + c * tau / (4.0 * r))
            assert 10 * np.log10(exact / approx) == pytest.approx(
                predicted, abs=1e-12)

    @pytest.mark.parametrize('call, beam, expected', [
        ('boundary_reverberation', 'horizontal_beamwidth_rad', 2.43),
        ('volume_reverberation', 'solid_angle_beamwidth_sr', 5.12),
    ])
    def test_the_warning_quotes_its_own_cell_geometry(self, call, beam,
                                                      expected):
        # D = c*tau/2 = 75 m at r = 50 m: the annulus is low by
        # 10*log10(1 + D/(2r)) = 2.43 dB and the shell by
        # 10*log10(1 + D/r + D**2/(3r**2)) = 5.12 dB.
        from uacpy.sonar import reverberation
        fn = getattr(reverberation, call)
        with pytest.warns(UserWarning,
                          match=f'low by about {expected:.2f} dB'):
            fn(np.array([50.0, 500.0]), 200.0, -27.0, pulse_length_s=0.1,
               sound_speed=1500.0, **{beam: 0.1})

    def test_the_shell_error_matches_the_exact_shell_volume(self):
        c, tau, psi = 1500.0, 0.1, 0.01
        extent = c * tau / 2.0
        for r in (10.0, 75.0, 150.0, 750.0):
            exact = psi * ((r + extent) ** 3 - r ** 3) / 3.0
            approx = psi * r ** 2 * extent
            ratio = extent / r
            predicted = 10 * np.log10(1.0 + ratio + ratio ** 2 / 3.0)
            assert 10 * np.log10(exact / approx) == pytest.approx(
                predicted, abs=1e-12)

    def test_the_far_field_is_silent_and_matches_the_closed_form(self):
        from uacpy.sonar.reverberation import boundary_reverberation
        r = np.array([500.0, 2000.0, 9000.0])
        with recorded_warnings() as rec:
            got = boundary_reverberation(r, 200.0, -27.0, **self.KW)
        expected = (200.0 - 2 * 20 * np.log10(r) - 27.0
                    + 10 * np.log10(0.1 * r * (1500.0 * 0.1 / 2.0)))
        np.testing.assert_allclose(got, expected)
        assert rec == []

    def test_a_cell_longer_than_the_range_warns_with_the_error(self):
        from uacpy.sonar.reverberation import boundary_reverberation
        with recorded_warnings() as rec:
            boundary_reverberation(np.array([1.0, 10.0, 1000.0]), 200.0,
                                   -27.0, **self.KW)
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert 'c*tau/2 = 75 m' in msg and 'shortest 1 m' in msg

    def test_volume_reverberation_guards_the_same_domain(self):
        from uacpy.sonar.reverberation import volume_reverberation
        kw = dict(pulse_length_s=0.1, solid_angle_beamwidth_sr=0.01,
                  sound_speed=1500.0)
        with recorded_warnings() as rec:
            volume_reverberation(np.array([1.0, 1000.0]), 200.0, -70.0, **kw)
        assert len(rec) == 1

    def test_a_zero_range_is_nan_without_a_numpy_warning(self):
        from uacpy.sonar.reverberation import boundary_reverberation
        with recorded_warnings() as rec:
            out = boundary_reverberation(np.array([0.0, 1000.0]), 200.0,
                                         -27.0, **self.KW)
        assert np.isnan(out[0]) and np.isfinite(out[1])
        assert [w for w in rec if issubclass(w.category, RuntimeWarning)] == []

    @pytest.mark.parametrize('fn,extra', [
        ('boundary_reverberation', {'horizontal_beamwidth_rad': 0.1}),
        ('volume_reverberation', {'solid_angle_beamwidth_sr': 0.01}),
    ])
    def test_a_negative_range_is_refused(self, fn, extra):
        from uacpy.sonar import reverberation
        with pytest.raises(ConfigurationError, match='must be >= 0'):
            getattr(reverberation, fn)(
                np.array([-100.0, 100.0]), 200.0, -27.0,
                pulse_length_s=0.1, sound_speed=1500.0, **extra)


class TestChapmanHarrisFlagsExtrapolation:
    """Chapman & Harris (1962) is a fit to measurements, not an approximation
    to a computable quantity, so there is no exact reference to bound its error
    against — the fitted envelope is the only information about how far the
    number can be trusted. It does not fail loudly either: the form stays
    smooth, monotone and physically plausible far outside the band, so an
    extrapolated value is indistinguishable from a validated one. At 10 kn and
    10 deg grazing it runs from -76.50 dB at 100 Hz to -28.76 dB at 200 kHz.

    Every bound here is quoted from the corpus, and the two sources disagree:

    * JKPS Sect. 1.7.1 — curves "derived from measurements over the frequency
      range of 400-6400 Hz and wind speed up to 15 m/s", and they "perform
      well for grazing angles below 40-50 deg, but fail to account for the
      high-angle roughness effects".
    * Etter Sect. 9.3.1 — "Chapman and Scott (1964) later validated these
      results over the frequency range 0.1 kHz to 6.4 kHz for grazing angle
      below 80 deg."

    The threshold follows JKPS, because that is the statement about the
    formula being RIGHT; Chapman & Scott's 80 deg is how far the data reach,
    which is a different claim and is only reported in the message.
    """

    def test_the_fitted_envelope_is_silent_and_matches_the_published_form(self):
        # Angles stay at or under the 40 deg the fit's data reached
        # (Abraham Sect. 3.5.3), and the wind under the 15 m/s it was
        # measured over.
        from uacpy.sonar.scattering import chapman_harris_surface
        angles = np.array([10.0, 30.0, 40.0])
        with recorded_warnings() as rec:
            got = chapman_harris_surface(frequency=1000.0,
                                         grazing_deg=angles, wind_speed_kn=10.0)
        assert rec == []
        # Independent transcription of the published form.
        beta = 158.0 * (10.0 * 1000.0 ** (1.0 / 3.0)) ** (-0.58)
        expected = (3.3 * beta * np.log10(angles / 30.0)
                    - 42.4 * np.log10(beta) + 2.6)
        np.testing.assert_allclose(got, expected)

    @pytest.mark.parametrize('freq', [100.0, 50000.0])
    def test_a_frequency_outside_the_fitted_band_warns(self, freq):
        from uacpy.sonar.scattering import chapman_harris_surface
        with recorded_warnings() as rec:
            chapman_harris_surface(frequency=freq,
                                   grazing_deg=10.0, wind_speed_kn=10.0)
        hits = [w for w in rec if 'outside the 400-6400 Hz' in str(w.message)]
        assert len(hits) == 1

    @pytest.mark.parametrize('freq', [400.0, 6400.0])
    def test_the_band_edges_are_inclusive(self, freq):
        from uacpy.sonar.scattering import chapman_harris_surface
        with recorded_warnings() as rec:
            chapman_harris_surface(frequency=freq,
                                   grazing_deg=10.0, wind_speed_kn=10.0)
        assert rec == []

    def test_a_grazing_angle_past_the_accuracy_limit_warns(self):
        from uacpy.sonar.scattering import chapman_harris_surface
        with recorded_warnings() as rec:
            chapman_harris_surface(frequency=1000.0,
                                   grazing_deg=np.array([10.0, 85.0]), wind_speed_kn=10.0)
        hits = [w for w in rec if 'grazing angle(s) exceed' in str(w.message)]
        assert len(hits) == 1
        msg = str(hits[0].message)
        assert 'steepest 85 deg' in msg
        # Both figures reach the caller, attributed to their own source.
        assert 'below 40-50 deg' in msg and '80 deg' in msg

    def test_the_first_angle_past_the_fitted_data_warns(self):
        from uacpy.sonar.scattering import chapman_harris_surface
        with recorded_warnings() as rec:
            chapman_harris_surface(frequency=1000.0, grazing_deg=40.5,
                                   wind_speed_kn=10.0)
        hits = [w for w in rec if 'grazing angle(s) exceed' in str(w.message)]
        assert len(hits) == 1
        assert 'below 40 deg (Abraham Sect. 3.5.3)' in str(hits[0].message)

    def test_the_threshold_is_the_jkps_accuracy_limit_not_the_data_range(self):
        # 60 deg is inside Chapman & Scott's 80 deg data range but past the
        # 40-50 deg JKPS says the formula performs well within.
        from uacpy.sonar.scattering import chapman_harris_surface
        with recorded_warnings() as rec:
            chapman_harris_surface(frequency=1000.0,
                                   grazing_deg=60.0, wind_speed_kn=10.0)
        assert [w for w in rec if 'grazing angle(s) exceed' in str(w.message)]

    def test_a_wind_speed_past_the_fitted_ceiling_warns(self):
        # JKPS Sect. 1.7.1 gives the fit a 15 m/s ceiling; nothing checked it.
        from uacpy.sonar.scattering import chapman_harris_surface
        with recorded_warnings() as rec:
            chapman_harris_surface(frequency=1000.0,
                                   grazing_deg=10.0, wind_speed_kn=40.0)      # 40 kn = 20.6 m/s
        hits = [w for w in rec if 'wind speed' in str(w.message)]
        assert len(hits) == 1 and '15 m/s' in str(hits[0].message)

    def test_a_wind_speed_inside_the_ceiling_is_silent(self):
        from uacpy.sonar.scattering import chapman_harris_surface
        with recorded_warnings() as rec:
            chapman_harris_surface(frequency=1000.0,
                                   grazing_deg=10.0, wind_speed_kn=25.0)      # 25 kn = 12.9 m/s
        assert [w for w in rec if 'wind speed' in str(w.message)] == []

    def test_the_two_envelope_checks_are_independent(self):
        from uacpy.sonar.scattering import chapman_harris_surface
        with recorded_warnings() as rec:
            chapman_harris_surface(frequency=50000.0,
                                   grazing_deg=85.0, wind_speed_kn=10.0)
        assert len(rec) == 2


@pytest.mark.parametrize('extra, phrase', [
    ({'target_strength_dB': 5.0}, 'target_strength_dB belongs to an active'),
    ({'target_strength_dB': 5.0, 'reverberation_level_dB': 70.0},
     'reverberation_level_dB belong to an active (echo) budget; a passive '
     'budget takes neither')])
def test_a_passive_budget_names_the_active_terms_it_was_given(extra, phrase):
    with pytest.raises(ConfigurationError, match='active') as err:
        sonar.SonarBudget('passive', source_level_dB=150.0,
                          noise_level_dB=60.0, detection_threshold_dB=10.0,
                          **extra)
    assert phrase in str(err.value)


@pytest.mark.parametrize('build', ['from_grain_size', 'literal'])
def test_bottom_parameters_hold_plain_floats(build):
    """from_grain_size computed loss_parameter in numpy and the repr printed
    np.float64(0.0161...) beside plain floats."""
    params = (sonar.BottomParameters.from_grain_size(2.0)
              if build == 'from_grain_size' else
              sonar.BottomParameters(np.float64(1.6), 1.14, np.float64(0.016),
                                     0.002, 0.0035))
    assert 'np.float64' not in repr(params)
    assert all(type(getattr(params, name)) is float
               for name in ('density_ratio', 'speed_ratio', 'loss_parameter',
                            'volume_parameter', 'spectral_strength',
                            'spectral_exponent'))


class TestBeamwidthAndLambertCeilings:
    """A beamwidth past the whole circle (2π rad) or sphere (4π sr) is a
    width typed in degrees: 12 'rad' returned 107.8 dB with no word. Lambert's
    mu is capped at 1/π (−4.97 dB) by energy conservation."""

    @pytest.mark.parametrize('call, beam, ceiling', [
        ('boundary_reverberation', 'horizontal_beamwidth_rad', 2.0 * np.pi),
        ('volume_reverberation', 'solid_angle_beamwidth_sr', 4.0 * np.pi)])
    @pytest.mark.parametrize('factor, refused', [(1.0, False),
                                                 (1.0 + 1e-9, True),
                                                 (3.0, True)])
    def test_a_beamwidth_past_every_direction_is_refused(
            self, call, beam, ceiling, factor, refused):
        kw = {'pulse_length_s': 0.01, beam: ceiling * factor}
        fn = getattr(sonar, call)
        if refused:
            with pytest.raises(ConfigurationError, match=f'{beam}=.* exceeds'):
                fn(np.array([1000.0]), 200.0, -30.0, **kw)
        else:
            assert np.isfinite(fn(np.array([1000.0]), 200.0, -30.0, **kw)[0])

    @pytest.mark.parametrize('mu_dB, warns', [
        (10 * np.log10(1 / np.pi), False), (10 * np.log10(1 / np.pi) + 1e-9,
                                            True), (0.0, True), (-27.0, False)])
    def test_a_lambert_coefficient_above_one_over_pi_warns(self, mu_dB, warns):
        msgs = warning_messages(lambda: sonar.lambert_bottom(30.0, mu_dB=mu_dB),
                                'mu = 1/pi')
        assert len(msgs) == (1 if warns else 0)


class TestLambertBottomFlagsExtrapolation:
    """Every constant here is quoted from the corpus.

    * Etter Sect. 9.3.3, citing Urick (1983) Ch. 8: Lambert's law "appears to
      provide a good approximation to the observed data for many deep-water
      bottoms at grazing angles below about 45 deg".
    * Etter Eq. 9.6, on Mackenzie's (1961) two-frequency deep-water
      measurements: "The term 10 log10 mu was found to be constant at -27 dB
      for both frequencies."
    * JKPS Sect. 1.7.2: for unconsolidated sediments the coefficient "assumes
      values between -25 and -35 dB", with "-29 dB a popular first guess".

    The 45 deg bound was documented in the docstring but never enforced, while
    the surface law in the same module warns at its own limit.
    """

    def test_the_default_coefficient_is_mackenzies_minus_27(self):
        from uacpy.sonar.scattering import LAMBERT_MU_DB
        assert LAMBERT_MU_DB == -27.0

    def test_the_default_sits_inside_the_jkps_sediment_spread(self):
        from uacpy.sonar.scattering import LAMBERT_MU_DB
        assert -35.0 <= LAMBERT_MU_DB <= -25.0

    def test_grazing_below_the_bound_is_silent_and_matches_etter_eq_9_6(self):
        from uacpy.sonar.scattering import LAMBERT_MU_DB, lambert_bottom
        angles = np.array([5.0, 20.0, 45.0])
        with recorded_warnings() as rec:
            got = lambert_bottom(angles)
        assert rec == []
        expected = LAMBERT_MU_DB + 10.0 * np.log10(
            np.sin(np.deg2rad(angles)) ** 2)
        np.testing.assert_allclose(got, expected)

    def test_a_steeper_angle_warns_and_names_the_source(self):
        from uacpy.sonar.scattering import lambert_bottom
        with recorded_warnings() as rec:
            lambert_bottom(np.array([10.0, 70.0]))
        hits = [w for w in rec if 'lambert_bottom' in str(w.message)]
        assert len(hits) == 1
        msg = str(hits[0].message)
        assert 'steepest 70 deg' in msg and 'about 45 deg' in msg

    def test_normal_incidence_returns_the_bare_coefficient(self):
        # sin(90 deg) = 1, so 10*log10(sin^2) vanishes and S_B == 10 log10 mu.
        from uacpy.sonar.scattering import LAMBERT_MU_DB, lambert_bottom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = float(np.atleast_1d(lambert_bottom(90.0))[0])
        assert got == pytest.approx(LAMBERT_MU_DB)


class TestAlbersheimFlagsExtrapolation:
    """Richards (2014) states Albersheim's eq. (1) is accurate to ~0.2 dB
    over ``0.1 <= P_D <= 0.9``, ``1e-7 <= P_F <= 1e-3`` and
    ``1 <= N <= 8096``; outside that envelope the value is an extrapolation
    of an empirical fit, and the function warns."""

    @pytest.mark.parametrize("pd, pf, n", [
        (0.5, 1e-4, 10),        # interior operating point
        (0.1, 1e-7, 8096),      # lower P_D/P_F edge, upper N edge
        (0.9, 1e-3, 1),         # upper P_D/P_F edge, lower N edge
    ])
    def test_inside_the_envelope_is_silent(self, pd, pf, n):
        with recorded_warnings() as rec:
            sonar.albersheim_snr(pd, pf, n)
        assert rec == []

    @pytest.mark.parametrize("pd, pf, n", [
        (0.95, 1e-4, 1),        # P_D above 0.9
        (0.05, 1e-4, 1),        # P_D below 0.1
        (0.5, 1e-8, 1),         # P_F below 1e-7
        (0.5, 1e-2, 1),         # P_F above 1e-3
        (0.5, 1e-4, 10000),     # N above 8096
    ])
    def test_outside_the_envelope_warns_citing_richards(self, pd, pf, n):
        with recorded_warnings() as rec:
            sonar.albersheim_snr(pd, pf, n)
        hits = [w for w in rec if 'albersheim_snr' in str(w.message)]
        assert len(hits) == 1
        assert issubclass(hits[0].category, UserWarning)
        msg = str(hits[0].message)
        assert 'Richards 2014' in msg
        assert '0.2 dB accuracy bound does not apply' in msg


class TestCsdmRequiresSnapshots:
    def test_zero_snapshot_columns_raise_a_typed_error(self):
        # simplefilter('error') turns the bare numpy divide RuntimeWarning of
        # an unguarded 0-column average into a failure, so the guard must
        # raise before the division.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ConfigurationError, match='zero snapshot'):
                sonar.csdm(np.zeros((4, 0)))

    def test_a_single_snapshot_column_yields_its_outer_product(self):
        d = np.array([[1.0 + 1.0j], [2.0 - 1.0j]])
        K = sonar.csdm(d)
        np.testing.assert_allclose(K, d @ d.conj().T)


class TestCsdmAndSampleCovarianceShareOneCore:
    """``csdm`` and ``acoustic_signal.sample_covariance`` compute the same
    ``(d dH)/L`` average, so they accept and refuse the same snapshots.

    A NaN snapshot is realistic: every engine NaNs the ``r <= 0`` columns of a
    point-source field and Bellhop NaNs shadow-zone cells, so snapshots
    assembled from modelled pressure carry them. One NaN makes every entry of
    ``K`` NaN, and ``bartlett`` scores an all-NaN ambiguity surface without
    raising anything at all.
    """

    @staticmethod
    def _snapshots():
        rng = np.random.default_rng(7)
        return (rng.standard_normal((6, 20))
                + 1j * rng.standard_normal((6, 20)))

    def test_the_two_names_return_the_same_matrix(self):
        from uacpy.acoustic_signal.beamforming import sample_covariance
        d = self._snapshots()
        np.testing.assert_array_equal(sonar.csdm(d), sample_covariance(d))

    @pytest.mark.parametrize('bad', [NAN, INF])
    def test_csdm_refuses_a_non_finite_snapshot(self, bad):
        d = self._snapshots()
        d[2, 3] = bad
        with pytest.raises(ConfigurationError, match='NaN or Inf'):
            sonar.csdm(d)

    @pytest.mark.parametrize('bad', [NAN, INF])
    def test_sample_covariance_refuses_a_non_finite_snapshot(self, bad):
        from uacpy.acoustic_signal.beamforming import sample_covariance
        d = self._snapshots()
        d[2, 3] = bad
        with pytest.raises(ConfigurationError, match='NaN or Inf'):
            sample_covariance(d)

    def test_the_non_finite_message_names_the_count_and_first_index(self):
        d = self._snapshots()
        d[2, 3] = NAN
        with pytest.raises(ConfigurationError,
                           match='snapshots contain NaN or Inf') as exc:
            sonar.csdm(d)
        message = str(exc.value)
        assert 'csdm:' in message
        assert '1 non-finite value(s) of 120' in message
        assert 'first at flat index 43' in message

    @pytest.mark.parametrize('shape, cue', [((0, 5), 'zero sensor rows'),
                                            ((6, 0), 'zero snapshot')])
    @pytest.mark.parametrize('name', ['csdm', 'sample_covariance'])
    def test_both_degenerate_axes_are_refused_on_both_entry_points(
            self, name, shape, cue):
        """A zero-sensor matrix averages to an empty ``(0, 0)`` covariance and
        every beamformer scores that as a silent all-zero surface — the same
        confident-meaningless-answer shape as the NaN case, on the other
        axis."""
        from uacpy.acoustic_signal.beamforming import sample_covariance
        call = sonar.csdm if name == 'csdm' else sample_covariance
        with pytest.raises(ConfigurationError, match=cue) as exc:
            call(np.zeros(shape, dtype=complex))
        assert name in str(exc.value)

    def test_a_zero_sensor_covariance_never_reaches_a_surface(self):
        from uacpy.acoustic_signal.beamforming import (
            bartlett, sample_covariance,
        )
        with pytest.raises(ConfigurationError,
                           match='snapshots has zero sensor rows'):
            bartlett(sample_covariance(np.zeros((0, 5), complex)),
                     np.zeros((3, 0), complex))

    @pytest.mark.parametrize('name', ['csdm', 'sample_covariance'])
    def test_a_field_is_named_with_the_bridge_rather_than_a_cast_error(
            self, name):
        """``np.asarray(Field, dtype=complex)`` raises ``must be real number,
        not Field``, which names nothing the caller passed. Both covariance
        names give the typed message the rest of the package gives."""
        from uacpy.acoustic_signal.beamforming import sample_covariance
        from uacpy.core.results import Field
        call = sonar.csdm if name == 'csdm' else sample_covariance
        t = np.arange(64) / 64.0
        field = Field(data=np.sin(2 * np.pi * 4 * t), coords={'time': t})
        with pytest.raises(
                ConfigurationError,
                match='snapshots must be a numeric array; got Field') as exc:
            call(field)
        message = str(exc.value)
        assert name in message
        assert 'Field.data' in message

    def test_a_one_sensor_one_snapshot_matrix_is_the_admissible_corner(self):
        """Both sides of both new boundaries: ``(1, 1)`` is the smallest
        matrix that still defines a covariance, and it is accepted."""
        from uacpy.acoustic_signal.beamforming import sample_covariance
        d = np.array([[2.0 + 1.0j]])
        np.testing.assert_allclose(sonar.csdm(d), d @ d.conj().T)
        np.testing.assert_allclose(sample_covariance(d), d @ d.conj().T)

    def test_an_all_finite_snapshot_set_beamforms(self):
        # The negative control for the guard: the ordinary path is untouched,
        # and the surface it produces carries no NaN.
        rng = np.random.default_rng(11)
        d = self._snapshots()
        surface = sonar.bartlett(sonar.csdm(d), self._replicas(rng))
        assert np.all(np.isfinite(surface.data))

    @staticmethod
    def _replicas(rng):
        """Five random replicas over the six-element array, one per
        candidate range."""
        from uacpy.core.results import Replicas
        rows = (rng.standard_normal((5, 6))
                + 1j * rng.standard_normal((5, 6)))
        return Replicas(replicas=rows[None],
                        candidates={'range': np.arange(1.0, 6.0)})

    def test_a_nan_snapshot_never_reaches_the_ambiguity_surface(self):
        rng = np.random.default_rng(11)
        d = self._snapshots()
        d[0, 0] = NAN
        replicas = self._replicas(rng)
        for processor in (sonar.bartlett, sonar.mvdr):
            with pytest.raises(ConfigurationError,
                               match='snapshots contain NaN or Inf'):
                processor(sonar.csdm(d), replicas)


class TestSonarPositivityGuardsRefuseNaN:
    @pytest.mark.parametrize("kwargs", [
        {'wind_speed_kn': NAN, 'frequency': 1000.0},
        {'wind_speed_kn': 10.0, 'frequency': NAN},
    ])
    def test_chapman_harris_nan_wind_or_frequency_raises(self, kwargs):
        with pytest.raises(ConfigurationError, match="must be > 0 and finite"):
            chapman_harris_surface(grazing_deg=10.0, **kwargs)

    def test_chapman_harris_nan_grazing_angle_warns(self):
        # The grazing-angle guard warns rather than raising (a 0 deg angle is a
        # legitimate -inf), so NaN must reach the same warning a negative does.
        with pytest.warns(UserWarning, match="negative or non-finite"):
            out = chapman_harris_surface(frequency=1000.0,
                                         grazing_deg=NAN, wind_speed_kn=10.0)
        assert np.isnan(out)

    def test_chapman_harris_zero_grazing_angle_stays_minus_inf_unwarned(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.isneginf(chapman_harris_surface(frequency=1000.0,
                                                      grazing_deg=0.0, wind_speed_kn=10.0))

    def test_column_scattering_nan_thickness_raises(self):
        with pytest.raises(ConfigurationError, match="must be > 0 and finite"):
            column_scattering_strength(-30.0, NAN)

    @pytest.mark.parametrize("call", [
        lambda: ts.ts_sphere(NAN),
        lambda: ts.ts_cylinder(NAN, 2.0, frequency=1000.0),
        lambda: ts.ts_plate(1.0, 1.0, frequency=NAN),
        lambda: ts.ts_ellipsoid(2.0, 1.0, NAN),
    ])
    def test_target_strength_nan_dimension_or_frequency_raises(self, call):
        with pytest.raises(ConfigurationError, match="must be > 0 and finite"):
            call()

    @pytest.mark.parametrize("call", [
        lambda: ts.ts_sphere(INF),
        lambda: ts.ts_cylinder(INF, 2.0, frequency=1000.0),
        lambda: ts.ts_plate(1.0, 1.0, frequency=INF),
        lambda: ts.ts_ellipsoid(2.0, 1.0, INF),
    ])
    def test_target_strength_infinite_dimension_or_frequency_raises(self, call):
        """The other half of "positive **and finite**", which the message
        promised and the guard did not deliver: ``inf > 0`` is True, so an
        infinite radius or frequency passed and every ts_* function returned an
        infinite target strength with no warning. NaN was refused because
        ``nan > 0`` is False — one negated comparison closed one hole and not
        the other."""
        with pytest.raises(ConfigurationError, match="must be > 0 and finite"):
            call()

    @pytest.mark.parametrize("bad", [None, [1.0, 2.0], np.array([1.0, 2.0]),
                                     object(), "abc"],
                             ids=['None', 'list', 'array', 'object', 'str'])
    def test_target_strength_non_numeric_is_a_typed_error(self, bad):
        """A non-numeric argument used to escape as the raw ``TypeError`` /
        ``ValueError`` ``float()`` raises, naming neither the function nor the
        parameter — a list of radii reported only "only length-1 arrays can be
        converted to Python scalars". The sibling guard in ``acoustic_signal``
        already typed this case; both now say which argument."""
        with pytest.raises(ConfigurationError,
                           match="radius_m must be a scalar number"):
            ts.ts_sphere(bad)

    def test_probability_of_detection_nan_pf_raises(self):
        with pytest.raises(
                ConfigurationError,
                match=r"pf must be in \(0, 1\)"):
            probability_of_detection(2.0, NAN)

    def test_roc_curve_nan_deflection_raises(self):
        with pytest.raises(ConfigurationError, match="must be >= 0 and finite"):
            roc_curve(NAN)

    @pytest.mark.parametrize("kwargs", [
        {'bandwidth_hz': NAN, 'integration_time_s': 1.0},
        {'bandwidth_hz': 100.0, 'integration_time_s': NAN},
    ])
    def test_detection_threshold_energy_nan_band_or_time_raises(self, kwargs):
        with pytest.raises(ConfigurationError, match="must be > 0 and finite"):
            detection_threshold_energy(0.9, 1e-4, **kwargs)


class TestThePositiveScalarGuardsAgreeAcrossLayers:
    """The package ships three "reject a non-positive scalar" guards, and that
    is deliberate — each names something different because each layer's user is
    asking a different question. The rationale lives in one place,
    ``_validate.require_positive``'s docstring; what lives here is the
    part prose cannot hold, which is that they agree on *what they accept*.

    They must, because they sit on the same values by different doors: a
    frequency handed to a carrier, to an estimator and to a target-strength
    formula is one frequency, and a layer that admits what another refuses is
    the shape of both bugs this split has already produced (``waveforms.py``'s
    copy, then the ``sonar`` one, each silently accepting ``inf`` against its
    own "and finite" message). A message may differ freely; an accept/reject
    verdict may not.

    Compared on behaviour rather than shape, so a future consolidation that
    unifies the *wording* and quietly changes a verdict fails here."""

    # Scalars only: the carrier guard is array-aware by design, so array input
    # is exactly where the three are *allowed* to differ.
    SCALARS = [
        ('zero', 0), ('zero float', 0.0), ('negative', -1),
        ('nan', NAN), ('inf', INF), ('-inf', -INF),
        ('one', 1.0), ('numpy float', np.float64(1.0)),
        ('numpy zero', np.float64(0.0)), ('numpy negative', np.float32(-1.0)),
        ('numpy int', np.int64(1)),
        ('0-d array', np.array(1.0)), ('0-d array zero', np.array(0.0)),
        ('True', True), ('False', False),
        ('numeric string', '1.0'), ('non-numeric string', 'abc'),
        ('None', None), ('object', object()),
    ]

    @staticmethod
    def _guards():
        from uacpy.core._validate import require_positive_finite_scalar
        from uacpy.core._validate import require_positive as carrier
        return {
            'core': lambda v: carrier(v, 'x'),
            'acoustic_signal': lambda v: require_positive_finite_scalar(
                v, 'caller', 'x'),
            'sonar': lambda v: ts._require_positive(v, 'x'),
        }

    @pytest.mark.parametrize('label,value', SCALARS,
                             ids=[label for label, _ in SCALARS])
    def test_the_three_guards_reach_the_same_verdict(self, label, value):
        verdicts = {}
        for name, guard in self._guards().items():
            try:
                guard(value)
                verdicts[name] = 'accept'
            except Exception:                                   # noqa: BLE001
                verdicts[name] = 'reject'
        assert len(set(verdicts.values())) == 1, verdicts

    @pytest.mark.parametrize('value', [0, -1.0, NAN, INF, -INF, None],
                             ids=['zero', 'negative', 'nan', 'inf', '-inf',
                                  'None'])
    def test_a_numeric_rejection_is_typed_in_all_three(self, value):
        """Agreeing on the verdict is not enough if one layer answers with an
        untyped exception: a caller writing ``except ConfigurationError``
        around a uacpy call would catch two of the three.

        ``None`` belongs here rather than with the non-numeric arguments
        below, and only measurement says so: ``np.asarray(None, dtype=float)``
        is ``array(nan)``, so the carrier guard refuses it as a NaN and types
        it like any other number."""
        for name, guard in self._guards().items():
            with pytest.raises(ConfigurationError, match='x must be'):
                guard(value)

    @pytest.mark.parametrize('value', ['abc', object()],
                             ids=['str', 'object'])
    def test_a_non_numeric_argument_is_typed_by_the_two_scalar_guards(
            self, value):
        """All three *reject* a non-numeric argument — the verdict agrees —
        but only the two scalar guards name it. ``sonar``'s used to leak the
        raw ``TypeError`` / ``ValueError`` that ``float()`` raises, and now
        wraps it as its sibling always did.

        The carrier guard deliberately still lets its conversion's own error
        through, and is excluded here rather than silently expected to change:
        it converts with ``np.asarray(..., dtype=float)``, where a failure is a
        dtype or ragged-shape problem rather than this guard's positivity
        verdict, and a carrier test already pins the ``ValueError`` a
        non-numeric ``SedimentLayer`` field raises. Making it typed is a
        deliberate breaking change, not a tidy-up, so it is recorded here and
        left."""
        from uacpy.core._validate import require_positive as carrier
        for name in ('acoustic_signal', 'sonar'):
            with pytest.raises(ConfigurationError,
                               match='x must be a scalar number'):
                self._guards()[name](value)
        with pytest.raises(
                (TypeError, ValueError),
                match=r'float\(\) argument must be|could not convert string to float'):
            carrier(value, 'x')

    def test_the_scalar_guards_return_the_value_as_a_float(self):
        """The two scalar guards are used as ``x = guard(x)``; the carrier one
        validates in place and returns ``None``. Pinned so a consolidation
        cannot swap one contract for the other silently."""
        from uacpy.core._validate import require_positive_finite_scalar
        assert require_positive_finite_scalar(
            np.float64(2.0), 'caller', 'x') == 2.0
        assert isinstance(ts._require_positive(np.float64(2.0), 'x'), float)


class TestReverberationGuardsRefuseNaNButKeepZeroRange:
    def test_boundary_reverberation_nan_pulse_length_raises(self):
        with pytest.raises(ConfigurationError, match="must be > 0 and finite"):
            boundary_reverberation([100.0, 200.0], 200.0, -30.0,
                                   pulse_length_s=NAN,
                                   horizontal_beamwidth_rad=0.1)

    def test_volume_reverberation_nan_solid_angle_raises(self):
        with pytest.raises(ConfigurationError, match="must be > 0 and finite"):
            volume_reverberation([100.0, 200.0], 200.0, -70.0,
                                 pulse_length_s=0.01,
                                 solid_angle_beamwidth_sr=NAN)

    @pytest.mark.parametrize("bad,first", [
        (dict(pulse_length_s=NAN, sound_speed=-1.0, ranges_m=[-1.0]),
         "pulse_length_s and horizontal_beamwidth_rad must be > 0 and finite"),
        (dict(sound_speed=-1.0, ranges_m=[-1.0]),
         "boundary_reverberation: sound_speed must be > 0 m/s and finite"),
        (dict(ranges_m=[-1.0]),
         "boundary_reverberation: ranges_m must be >= 0 and finite"),
    ])
    def test_boundary_reverberation_reports_the_first_failing_guard(
            self, bad, first):
        # Guard order is pulse/beam, sound speed, ranges: a call with several
        # bad arguments names the first of them.
        kw = dict(ranges_m=[100.0], pulse_length_s=0.01,
                  horizontal_beamwidth_rad=0.1, sound_speed=1500.0)
        kw.update(bad)
        with pytest.raises(ConfigurationError, match=first):
            boundary_reverberation(kw.pop("ranges_m"), 200.0, -30.0, **kw)

    def test_volume_reverberation_names_its_own_beam_argument_first(self):
        with pytest.raises(ConfigurationError,
                           match="volume_reverberation: pulse_length_s and "
                                 "solid_angle_beamwidth_sr must be > 0 and "
                                 "finite; got pulse_length_s=0.01, "
                                 "solid_angle_beamwidth_sr=-0.1"):
            volume_reverberation([-1.0], 200.0, -70.0, pulse_length_s=0.01,
                                 solid_angle_beamwidth_sr=-0.1,
                                 sound_speed=0.0)

    # ``INF`` completes the set the test's name already claimed: the guard was
    # a bare ``~(r >= 0)``, which refuses NAN and -inf but admits +inf.
    @pytest.mark.parametrize("bad", [NAN, -np.inf, INF])
    def test_non_finite_range_raises(self, bad):
        with pytest.raises(ConfigurationError, match="must be >= 0 and finite"):
            boundary_reverberation([bad, 200.0], 200.0, -30.0,
                                   pulse_length_s=0.01,
                                   horizontal_beamwidth_rad=0.1)

    def test_zero_range_returns_nan_without_a_numpy_warning(self):
        # r == 0 is the package's no-data convention for a zero-area cell and
        # must keep passing the guard the NaN now fails.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = boundary_reverberation([0.0, 200.0], 200.0, -30.0,
                                         pulse_length_s=0.01,
                                         horizontal_beamwidth_rad=0.1)
        assert np.isnan(out[0]) and np.isfinite(out[1])


#: Every guarded parameter in ``sonar.reverberation``, ``sonar.detection`` and
#: ``sonar.scattering`` whose message promises "and finite", each call setting
#: exactly that one parameter to +inf and leaving the rest valid.
_INFINITY_SITES = [
    ('boundary_reverberation/ranges_m',
     lambda: boundary_reverberation([INF, 200.0], 200.0, -30.0,
                                    pulse_length_s=0.01,
                                    horizontal_beamwidth_rad=0.1)),
    ('boundary_reverberation/pulse_length_s',
     lambda: boundary_reverberation([100.0, 200.0], 200.0, -30.0,
                                    pulse_length_s=INF,
                                    horizontal_beamwidth_rad=0.1)),
    ('boundary_reverberation/horizontal_beamwidth_rad',
     lambda: boundary_reverberation([100.0, 200.0], 200.0, -30.0,
                                    pulse_length_s=0.01,
                                    horizontal_beamwidth_rad=INF)),
    ('volume_reverberation/ranges_m',
     lambda: volume_reverberation([INF, 200.0], 200.0, -70.0,
                                  pulse_length_s=0.01,
                                  solid_angle_beamwidth_sr=0.01)),
    ('volume_reverberation/pulse_length_s',
     lambda: volume_reverberation([100.0, 200.0], 200.0, -70.0,
                                  pulse_length_s=INF,
                                  solid_angle_beamwidth_sr=0.01)),
    ('volume_reverberation/solid_angle_beamwidth_sr',
     lambda: volume_reverberation([100.0, 200.0], 200.0, -70.0,
                                  pulse_length_s=0.01,
                                  solid_angle_beamwidth_sr=INF)),
    ('roc_curve/deflection', lambda: roc_curve(INF)),
    ('detection_threshold_energy/bandwidth_hz',
     lambda: detection_threshold_energy(0.9, 1e-4, INF, 1.0)),
    ('detection_threshold_energy/integration_time_s',
     lambda: detection_threshold_energy(0.9, 1e-4, 100.0, INF)),
    ('chapman_harris_surface/wind_speed_kn',
     lambda: chapman_harris_surface(frequency=1000.0,
                                    grazing_deg=10.0, wind_speed_kn=INF)),
    ('chapman_harris_surface/frequency',
     lambda: chapman_harris_surface(frequency=INF,
                                    grazing_deg=10.0, wind_speed_kn=10.0)),
    ('column_scattering_strength/thickness_m',
     lambda: column_scattering_strength(-30.0, INF)),
]


class TestSonarGuardsRefuseInfinity:
    """The other half of every "and finite" the sonar guards promise.

    Each of these guards was written as the negation of its admissible
    condition so NaN would be refused — ``nan > 0`` is False — and that single
    change closed the NaN hole while leaving the infinity one open, because
    ``inf > 0`` is True. Nothing in the message marks the difference: all of
    them already said "and finite", so the message was the lie, not the
    documentation.

    Measured before the fix, every site below returned instead of raising:
    both reverberation functions an infinite level, ``detection_threshold_energy``
    a ``DT`` of -inf (no signal at all required), ``column_scattering_strength``
    and ``chapman_harris_surface`` +inf, and the range guard a NaN that was
    indistinguishable from the deliberate zero-range one it exists to keep
    distinct. ``roc_curve`` is the site that argues for driving the guard
    rather than screening the output: an infinite deflection returns a
    perfectly finite curve, ``P_D == 1`` at every ``P_F``, so no finiteness
    check downstream would ever have flagged it.

    Parametrised over the sites because the defect is a class, not six
    accidents — this is the test that fails for the seventh guard written as a
    bare sign test behind a finiteness message.
    """

    @pytest.mark.parametrize(
        'call', [c for _, c in _INFINITY_SITES],
        ids=[name for name, _ in _INFINITY_SITES])
    def test_an_infinite_argument_raises_naming_finiteness(self, call):
        with pytest.raises(ConfigurationError, match='and finite'):
            call()

    def test_chapman_harris_infinite_grazing_angle_warns(self):
        """The grazing-angle site is the exception that warns rather than
        raising, because a 0 deg angle is a legitimate -inf. Its message
        already named "non-finite" while ``~(theta >= 0)`` admitted +inf, so
        the one angle that returns +inf rather than NaN was also the one that
        said nothing at all."""
        with pytest.warns(UserWarning, match='negative or non-finite'):
            out = chapman_harris_surface(frequency=1000.0,
                                         grazing_deg=INF, wind_speed_kn=10.0)
        assert not np.isfinite(out)

    @pytest.mark.parametrize('bad', [INF, -INF, NAN, 0.0, -1500.0],
                             ids=['inf', '-inf', 'nan', 'zero', 'negative'])
    @pytest.mark.parametrize('fn,extra', [
        ('boundary_reverberation', {'horizontal_beamwidth_rad': 0.1}),
        ('volume_reverberation', {'solid_angle_beamwidth_sr': 0.01}),
    ])
    def test_a_bad_sound_speed_is_refused(self, fn, extra, bad):
        """``sound_speed`` carried no check at all, and failed three ways in
        silence: ``inf`` gave an infinite level, ``0`` a -inf one (the cell
        collapses to zero area), and a negative or non-finite one a NaN.

        ``target_strength`` already refused all five of these values for the
        same quantity through ``_require_positive``; this door disagreed with
        that one, which is what the cross-layer guard-agreement test in this
        file exists to catch."""
        from uacpy.sonar import reverberation
        with pytest.raises(ConfigurationError,
                           match='sound_speed must be > 0 m/s and finite'):
            getattr(reverberation, fn)([100.0, 200.0], 200.0, -30.0,
                                       pulse_length_s=0.01, sound_speed=bad,
                                       **extra)

    @pytest.mark.parametrize('bad', [INF, -INF, NAN], ids=['inf', '-inf', 'nan'])
    def test_probability_of_detection_non_finite_deflection_raises(self, bad):
        """``pf`` was already refused for returning a silent NaN P_D; the same
        silent NaN arrived through ``deflection``. ``inf``/``-inf`` are the
        subtler half — they return exactly 1.0 and 0.0, valid-looking
        probabilities that no finiteness check downstream would flag."""
        with pytest.raises(ConfigurationError,
                           match='deflection must be finite'):
            probability_of_detection(bad, np.array([1e-4]))

    @pytest.mark.parametrize('bad', [INF, -INF, NAN, -10.0],
                             ids=['inf', '-inf', 'nan', 'negative'])
    def test_lambert_bottom_negative_or_non_finite_grazing_warns(self, bad):
        """The bottom law now says what the surface law already said. A -10 deg
        angle used to carry only numpy's anonymous "invalid value encountered",
        naming neither the function nor the argument, and a NaN angle carried
        nothing at all."""
        with pytest.warns(UserWarning, match='negative or non-finite'):
            out = sonar.lambert_bottom(np.array([bad]))
        assert not np.all(np.isfinite(out))

    def test_lambert_bottom_zero_grazing_stays_minus_inf_unwarned(self):
        """The documented degenerate answer, and the reason the grazing guard
        warns instead of raising — pinned so closing the guard cannot close
        this too."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert np.isneginf(sonar.lambert_bottom(np.array([0.0]))[0])

    @pytest.mark.parametrize('level', [-INF, NAN], ids=['-inf', 'nan'])
    def test_a_dB_level_keeps_its_non_finite_meaning(self, level):
        """The deliberate leaves, pinned so a later round cannot "harmonise"
        them into rejections.

        ``-inf`` dB is exactly zero linear power — an absent contribution — and
        ``total_reverberation`` relies on it: a -inf component adds nothing to
        the incoherent sum, where a 0 dB one shifts it. NaN is the package's
        own no-data convention, which ``boundary_reverberation`` deliberately
        *returns* at zero range. A finiteness guard on a dB-valued parameter
        would refuse both."""
        out = sonar.total_reverberation(np.array([level]), np.array([50.0]))
        if np.isneginf(level):
            assert out[0] == pytest.approx(50.0)
            assert sonar.total_reverberation(
                np.array([0.0]), np.array([50.0]))[0] > 50.0
        else:
            assert np.isnan(out[0])
        for dB_arg in (lambda v: sonar.boundary_reverberation(
                           [100.0, 200.0], v, -30.0, pulse_length_s=0.01,
                           horizontal_beamwidth_rad=0.1),
                       lambda v: sonar.column_scattering_strength(v, 10.0)):
            assert not np.all(np.isfinite(np.asarray(dB_arg(level), float)))

    def test_the_finite_path_is_untouched(self):
        """Rejecting infinity must not be bought by rejecting anything else:
        the legitimate degenerate inputs each guard was built around — a
        zero-area range cell, a 0 deg grazing angle, a zero deflection — still
        pass, and a wholly ordinary call still returns its finite level."""
        rl = boundary_reverberation([0.0, 100.0, 1000.0], 200.0, -27.0,
                                    pulse_length_s=0.01,
                                    horizontal_beamwidth_rad=0.1)
        assert np.isnan(rl[0]) and np.all(np.isfinite(rl[1:]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert np.isneginf(chapman_harris_surface(frequency=1000.0,
                                                      grazing_deg=0.0, wind_speed_kn=10.0))
            assert np.isfinite(chapman_harris_surface(frequency=1000.0,
                                                      grazing_deg=10.0, wind_speed_kn=10.0))
        assert np.all(np.isfinite(roc_curve(0.0)[1]))
        assert np.isfinite(detection_threshold_energy(0.9, 1e-4, 100.0, 1.0))
        assert np.isfinite(column_scattering_strength(-30.0, 10.0))


# ── detection_range far-edge recovery ───────────────────────────────────────
#
# `detection_range` returns inf when the signal excess is >= 0 at the far edge:
# the outermost crossing lies beyond the grid, and the grid's own edge would be
# a number that moves with `receiver.ranges` (measured on a Kraken shelf
# budget: 20000.0 m on a 20 km grid against 45947 m on a 120 km grid). When
# SE went negative inside the grid first, a warning names that shadow zone,
# since the inf is then not "detectable at every range".
#
# Every threshold below is pinned on BOTH sides: a mutation campaign found that
# a guard test using values far from the boundary pins that the guard fires,
# never where.

_SHADOW = 'NOT detectable at every range'


class TestDetectionRangeFarEdgeRecovery:
    def test_far_edge_recovery_warns_and_names_the_in_grid_shadow(self):
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        se = np.array([5.0, -1.0, 2.0, 3.0, 4.0])
        msgs = warning_messages(lambda: sonar.detection_range(r, signal_excess_dB=se), _SHADOW)
        assert len(msgs) == 1
        # SE crosses down at 5/6 and back up at 1 + 1/3.
        assert 'between 0.833333 and 1.33333 m' in msgs[0]
        assert 'detection_annuli' in msgs[0]
        assert "crossing='first'" in msgs[0]

    def test_a_shadow_at_the_nearest_range_is_the_one_named(self):
        r = np.array([0.0, 1.0, 2.0, 3.0])
        se = np.array([-1.0, -2.0, 1.0, 2.0])
        msgs = warning_messages(lambda: sonar.detection_range(r, signal_excess_dB=se), _SHADOW)
        assert len(msgs) == 1
        assert 'between 0 and 1.66667 m' in msgs[0]

    @pytest.mark.parametrize('n_shadowed', [0, 1, 2])
    def test_a_sweep_warns_once_whatever_the_number_of_shadowed_slices(
            self, n_shadowed):
        """One call is one warning: a 25-depth map gave 16 multi-line
        warnings, one per row."""
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        recovers = np.array([5.0, -1.0, 2.0, 3.0, 4.0])
        clean = np.array([5.0, 4.0, 3.0, -2.0, -3.0])
        rows = np.array([recovers] * n_shadowed + [clean] * (3 - n_shadowed))
        msgs = warning_messages(
            lambda: sonar.detection_ranges(r, signal_excess_dB=rows), _SHADOW)
        if n_shadowed == 0:
            assert msgs == []
        else:
            assert len(msgs) == 1
            assert f'{n_shadowed} of 3 slices' in msgs[0]
            assert 'between 0.833333 and 1.33333 m' in msgs[0]

    def test_the_depth_map_warning_names_the_first_shadowed_depth(self):
        from uacpy.core.results import Field
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        se = Field(data=np.array([[5.0, 4.0, 3.0, -2.0, -3.0],
                                  [5.0, -1.0, 2.0, 3.0, 4.0],
                                  [5.0, -1.0, 2.0, 3.0, 4.0]]),
                   coords={'depth': np.array([10.0, 20.0, 30.0]), 'range': r},
                   kind='signal_excess')
        msgs = warning_messages(lambda: sonar.detection_ranges_by_depth(se),
                                _SHADOW)
        assert len(msgs) == 1
        assert msgs[0].startswith('detection_ranges_by_depth: 2 of 3 rows')
        assert 'at depth 20:' in msgs[0]

    def test_far_edge_recovery_returns_inf(self):
        """The same inf as SE >= 0 everywhere: in both the outermost
        crossing lies beyond the grid. A convergence-zone shape (+, -, +)
        ending positive returns inf, not the inner down-crossing and not
        the grid's edge."""
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        se = np.array([5.0, -1.0, 2.0, 3.0, 4.0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert sonar.detection_range(r, signal_excess_dB=se) == np.inf

    def test_negative_at_the_last_sample_is_silent(self):
        """One sample either side of the branch: SE at the outermost range is
        -1 dB here and +1 dB in the sibling test, and only the second puts
        the crossing beyond the grid."""
        r = np.array([0.0, 1.0, 2.0, 3.0])
        se = np.array([5.0, -1.0, 1.0, -1.0])
        msgs = warning_messages(lambda: sonar.detection_range(r, signal_excess_dB=se), _SHADOW)
        assert msgs == []

    def test_positive_at_the_last_sample_warns(self):
        r = np.array([0.0, 1.0, 2.0, 3.0])
        se = np.array([5.0, -1.0, 1.0, 1.0])
        msgs = warning_messages(lambda: sonar.detection_range(r, signal_excess_dB=se), _SHADOW)
        assert len(msgs) == 1

    def test_zero_at_the_last_sample_warns(self):
        """``positive`` is ``se >= 0``, so SE == 0 at the far edge is the
        warning side of that comparison, not the silent one."""
        r = np.array([0.0, 1.0, 2.0, 3.0])
        se = np.array([5.0, -1.0, 1.0, 0.0])
        msgs = warning_messages(lambda: sonar.detection_range(r, signal_excess_dB=se), _SHADOW)
        assert len(msgs) == 1

    def test_positive_everywhere_is_silent(self):
        """``inf`` already says "beyond the grid"; the adjacent case is the
        same vector with one sample pushed negative, which does warn."""
        r = np.linspace(1.0, 10.0, 10)
        always = np.ones(10)
        assert warning_messages(lambda: sonar.detection_range(r, signal_excess_dB=always),
                         _SHADOW) == []
        dipped = always.copy()
        dipped[4] = -1.0
        assert len(warning_messages(lambda: sonar.detection_range(r, signal_excess_dB=dipped),
                             _SHADOW)) == 1

    def test_negative_everywhere_is_silent(self):
        r = np.linspace(1.0, 10.0, 10)
        assert warning_messages(lambda: sonar.detection_range(r, signal_excess_dB=-np.ones(10)),
                         _SHADOW) == []

    def test_no_data_hole_before_a_negative_far_edge_is_silent(self):
        r = np.array([0.0, 1000.0, 2000.0])
        se = np.array([5.0, np.nan, -5.0])
        assert warning_messages(lambda: sonar.detection_range(r, signal_excess_dB=se),
                         _SHADOW) == []

    def test_trailing_no_data_after_a_recovery_warns_about_the_last_filled_cell(self):
        """The masked array's last entry is the outermost cell the model
        FILLED, so the warning names that range, not the grid's edge."""
        r = np.array([0.0, 1.0, 2.0, 3.0])
        se = np.array([5.0, -1.0, 2.0, np.nan])
        msgs = warning_messages(lambda: sonar.detection_range(r, signal_excess_dB=se), _SHADOW)
        assert len(msgs) == 1
        assert '(2 m)' in msgs[0]


def _flat_tl_field():
    """A 5x7 depth-range Field holding 60 everywhere."""
    from uacpy.core.results import Field
    return Field(data=np.full((5, 7), 60.0),
                 coords={'depth': np.linspace(0.0, 100.0, 5),
                         'range': np.linspace(100.0, 5000.0, 7)})


class TestSignalExcessFieldRejectsANonScalarNoiseLevel:
    """``reverberation_level_dB`` is the one per-range term of this budget; the
    rest are documented scalars. An array ``noise_level_dB`` computed the whole
    signal-excess field and *then* raised ``TypeError: only 0-dimensional
    arrays can be converted to Python scalars`` from the budget dict, naming
    neither the function nor the argument."""

    @pytest.mark.parametrize('bad', ['per_range', 'two_d'])
    @pytest.mark.parametrize('mode', ['passive', 'active'])
    def test_an_array_noise_level_is_named_up_front(self, mode, bad):
        field = _flat_tl_field()
        nl = (np.linspace(55.0, 65.0, 7) if bad == 'per_range'
              else np.zeros((5, 7)))
        kwargs = ({'source_level_dB': 200.0} if mode == 'passive'
                  else {'source_level_dB': 200.0, 'target_strength_dB': -20.0})
        call = getattr(sonar, f'{mode}_signal_excess_field')
        with pytest.raises(
                ConfigurationError,
                match='noise_level_dB must be a scalar dB level') as exc:
            call(field, noise_level_dB=nl, detection_threshold_dB=0.0,
                 **kwargs)
        message = str(exc.value)
        assert f'{mode}_signal_excess_field' in message
        assert 'noise_level_dB' in message
        assert 'reverberation_level_dB' in message

    def test_a_scalar_noise_level_is_accepted(self):
        se = sonar.passive_signal_excess_field(
            _flat_tl_field(), source_level_dB=200.0, noise_level_dB=60.0,
            detection_threshold_dB=0.0)
        assert se.data.shape == (5, 7)

    @pytest.mark.parametrize('mode, label', [
        (mode, label) for mode in ('passive', 'active')
        for label in ('source_level_dB', 'directivity_index_dB',
                      'detection_threshold_dB', 'processing_loss_dB',
                      'target_strength_dB')
        if not (mode == 'passive' and label == 'target_strength_dB')])
    def test_every_scalar_budget_term_is_named_up_front(self, mode, label):
        kwargs = {'source_level_dB': 200.0, 'noise_level_dB': 60.0,
                  'detection_threshold_dB': 0.0}
        if mode == 'active':
            kwargs['target_strength_dB'] = -20.0
        kwargs[label] = np.array([1.0, 2.0, 3.0])
        call = getattr(sonar, f'{mode}_signal_excess_field')
        with pytest.raises(ConfigurationError,
                           match=f'{mode}_signal_excess_field: {label} must '
                                 f'be a scalar dB level'):
            call(_flat_tl_field(), **kwargs)

    def test_the_per_range_term_the_message_points_at_accepts_an_array(
            self):
        se = sonar.active_signal_excess_field(
            _flat_tl_field(), source_level_dB=200.0, target_strength_dB=-20.0,
            noise_level_dB=60.0,
            reverberation_level_dB=np.linspace(55.0, 65.0, 7), detection_threshold_dB=0.0)
        assert se.data.shape == (5, 7)


class TestScalarSonarEquationNamesItsFieldTwin:
    """Handing a ``Field`` to the array-taking sonar-equation functions
    reached ``float()`` and raised ``TypeError: float() argument must be a
    string or a real number, not 'Field'`` — while the working call is one
    suffix away and the message never said so."""

    def test_passive_signal_excess_names_passive_signal_excess_field(self):
        with pytest.raises(ConfigurationError,
                           match='Use passive_signal_excess_field') as exc:
            sonar.passive_signal_excess(180.0, _flat_tl_field(), 60.0, detection_threshold_dB=0.0)
        message = str(exc.value)
        assert 'passive_signal_excess_field' in message

    def test_active_signal_excess_names_active_signal_excess_field(self):
        with pytest.raises(ConfigurationError,
                           match='Use active_signal_excess_field') as exc:
            sonar.active_signal_excess(180.0, _flat_tl_field(), -20.0,
                                       noise_level_dB=60.0, detection_threshold_dB=0.0)
        assert 'active_signal_excess_field' in str(exc.value)

    def test_the_twin_the_message_names_accepts_the_same_field(self):
        se = sonar.passive_signal_excess_field(
            _flat_tl_field(), source_level_dB=180.0, noise_level_dB=60.0,
            detection_threshold_dB=0.0)
        assert se.data.shape == (5, 7)

    def test_a_plain_array_reaches_the_scalar_function(self):
        out = sonar.passive_signal_excess(180.0, np.full((5, 7), 60.0), 60.0,
                                          detection_threshold_dB=0.0)
        assert np.asarray(out).shape == (5, 7)


class TestScalarBudgetsRefuseAFieldWithUsableAdvice:
    """The scalar sonar-equation functions take dB arrays. Handed a pressure
    Field they name the ``*_field`` twin and the one array form that is a
    TL: ``.dB`` is a property, and ``.data`` is complex pressure, so neither
    ``.dB()`` nor ``.data`` may be advised."""

    @staticmethod
    def _pressure_field():
        from uacpy.core.results import Field
        tl_dB = np.array([[67.0, 50.0]])
        return Field(
            data=(10.0 ** (-tl_dB / 20.0)).astype(complex),
            coords={'depth': np.array([10.0]),
                    'range': np.array([1000.0, 2000.0])},
            model='Test', frequencies=100.0)

    @pytest.mark.parametrize('fn, twin', [
        (sonar.passive_signal_excess, 'passive_signal_excess_field'),
    ])
    def test_the_advice_names_the_twin_and_the_dB_property(self, fn, twin):
        f = self._pressure_field()
        with pytest.raises(ConfigurationError,
                           match='tl_dB is a Field') as info:
            fn(160.0, f, 60.0, 0.0, detection_threshold_dB=0.0)
        text = str(info.value)
        assert twin in text
        assert 'field.dB' in text and 'as tl_dB' in text
        assert '.dB()' not in text
        assert '.data' not in text

    def test_following_the_advice_gives_the_level_budget(self):
        f = self._pressure_field()
        se = sonar.passive_signal_excess(160.0, np.asarray(f.dB, float),
                                         60.0, 0.0, detection_threshold_dB=0.0)
        np.testing.assert_allclose(se, [[33.0, 50.0]], atol=1e-6)


# ── The APL-UW TR 9407 seabed and sea-surface scattering models ──────────────
# The bottom model is pinned to the report's own Table 3 (scattering strengths
# for generic bottom types, computed by its authors at 1528 m/s); the forward
# loss to its Figure 1; the parameter constructors to Table 2 and Eqs. 2-10;
# the surface model to the levels its Figures 3-8 show against data.

ANGLES = np.array([1, 2, 3, 5, 7, 10, 20, 40, 60, 70, 80, 85, 88, 89, 90.0])

#: TR 9407 Table 3 (pp. IV-23..25), the 30 kHz column unless stated.
TABLE_3 = {
    ('rough rock', 30e3): [-23.0, -19.4, -17.4, -14.8, -13.3, -11.3, -7.9, -5.4,
                           -4.9, -5.0, -5.4, -5.3, -5.4, -5.4, -5.4],
    ('rock', 30e3): [-44.0, -37.3, -33.3, -28.4, -25.2, -21.7, -15.4, -10.0,
                     -9.1, -8.3, -7.3, -6.8, -6.6, -6.6, -6.6],
    ('cobble', 30e3): [-48.5, -41.7, -37.6, -32.6, -29.2, -25.7, -19.3, -14.9,
                       -13.1, -11.5, -9.2, -8.2, -7.9, -7.9, -7.8],
    ('sandy gravel', 30e3): [-50.3, -44.2, -40.3, -35.1, -31.7, -28.1, -22.1,
                             -18.9, -16.5, -13.7, -9.9, -8.4, -8.0, -7.9, -7.8],
    ('coarse sand', 30e3): [-47.1, -43.8, -41.5, -37.9, -34.9, -31.5, -25.1,
                            -21.4, -18.8, -15.3, -8.9, -5.6, -4.6, -4.4, -4.4],
    ('medium sand', 10e3): [-61.5, -56.8, -53.2, -47.6, -43.4, -38.8, -30.9,
                            -24.4, -22.6, -20.4, -13.2, -5.0, 1.2, 2.3, 2.6],
    ('medium sand', 30e3): [-51.4, -48.1, -45.8, -42.1, -39.0, -35.4, -28.3,
                            -23.5, -21.2, -17.7, -10.8, -5.5, -3.4, -3.1, -3.1],
    ('medium sand', 100e3): [-50.0, -44.9, -41.5, -36.6, -33.3, -29.8, -24.2,
                             -21.7, -19.5, -16.4, -12.2, -10.4, -9.8, -9.7, -9.7],
    ('very fine sand', 30e3): [-59.7, -54.7, -50.8, -44.6, -40.1, -35.2, -27.1,
                               -24.9, -23.7, -23.1, -20.2, -13.1, -5.9, -4.3, -3.8],
    ('very fine sand', 100e3): [-51.4, -47.9, -45.4, -41.3, -37.9, -34.0, -27.0,
                                -24.8, -23.5, -22.3, -18.0, -12.9, -10.3, -10.0,
                                -10.0],
}
#: Table 3's two silt columns: Mz = 6.0 at sigma2 = 0.001 and 0.0003, 30 kHz.
TABLE_3_SILT = {
    0.001: [-58.2, -51.1, -46.6, -41.3, -38.3, -35.7, -31.8, -28.8, -27.4,
            -27.0, -26.3, -23.9, -13.1, -6.8, -3.6],
    0.0003: [-63.4, -56.3, -51.8, -46.5, -43.5, -40.9, -37.0, -34.0, -32.6,
             -32.1, -30.5, -25.8, -13.3, -6.8, -3.6],
}


class TestBottomBackscatterReproducesTable3:

    @pytest.mark.parametrize('name,freq', sorted(TABLE_3))
    def test_generic_bottom_column(self, name, freq):
        """Every column to 0.6 dB (the table prints 0.1 dB): the
        composite-roughness, large-roughness and volume terms to 0.3 dB,
        rock's 40 deg cell 0.6 dB, everything on sand within 0.25 dB."""
        got = apl_uw_bottom_backscatter(
            ANGLES, freq, BottomParameters.from_sediment(name),
            water_sound_speed=TABLE_WATER_SOUND_SPEED)
        d = np.abs(got - np.array(TABLE_3[(name, freq)]))
        assert d.max() < 0.6, f"{name} at {freq/1e3:g} kHz: {np.round(d, 2)}"

    @pytest.mark.parametrize('sigma2', sorted(TABLE_3_SILT))
    def test_silt_columns_follow_the_volume_parameter(self, sigma2):
        """The soft-bottom column moves with sigma2 alone (IV.C.3): the
        two silt tables differ only in that parameter."""
        p = BottomParameters.from_grain_size(6.0).replace(volume_parameter=sigma2)
        got = apl_uw_bottom_backscatter(ANGLES, 30e3, p,
                                        water_sound_speed=TABLE_WATER_SOUND_SPEED)
        assert np.abs(got - np.array(TABLE_3_SILT[sigma2])).max() < 0.25

    def test_kirchhoff_level_is_the_documented_factor_above_the_printed_form(self):
        """The near-vertical value would sit 3.6 dB under Table 3 without
        the level factor; the factor is 2^(2(1-alpha)/alpha) = 2.30 at
        gamma = 3.25, applied to the Kirchhoff branch only."""
        from uacpy.sonar.bottom_scattering import _kirchhoff_level
        assert _kirchhoff_level(3.25 / 2.0 - 1.0) == pytest.approx(2.0 ** 1.2)

    def test_zero_grazing_uses_the_value_at_one_thousandth_of_a_degree(self):
        p = BottomParameters.from_sediment('medium sand')
        s0, s1 = apl_uw_bottom_backscatter([0.0, 0.001], 30e3, p,
                                           water_sound_speed=1528.0)
        assert np.isfinite(s0) and s0 == s1

    def test_out_of_band_frequency_warns_and_bad_inputs_raise(self):
        p = BottomParameters.from_sediment('medium sand')
        with pytest.warns(UserWarning, match='10-100 kHz'):
            apl_uw_bottom_backscatter(20.0, 5e3, p)
        with pytest.warns(UserWarning, match='water_sound_speed'):
            apl_uw_bottom_backscatter(20.0, 30e3, p, water_sound_speed=1700.0)
        with pytest.raises(ConfigurationError, match='0-90'):
            apl_uw_bottom_backscatter(95.0, 30e3, p)
        with pytest.raises(ConfigurationError, match='frequency'):
            apl_uw_bottom_backscatter(20.0, np.nan, p)


class TestBottomForwardLoss:

    def test_slow_silt_shows_the_intromission_peak_of_figure_1(self):
        """Figure 1: silt (Mz = 6, nu < 1) peaks near 35 dB at about 16 deg
        and settles near 24 dB at vertical; rock loses under 0.1 dB below
        its 66 deg critical angle and about 2.8 dB at vertical."""
        silt = apl_uw_bottom_loss(np.arange(1.0, 90.5, 0.5),
                                  BottomParameters.from_grain_size(6.0))
        i = int(np.argmax(silt))
        assert 14.0 <= np.arange(1.0, 90.5, 0.5)[i] <= 18.0
        assert 33.0 < silt[i] < 37.0
        assert 23.0 < silt[-1] < 25.5
        rock = apl_uw_bottom_loss([30.0, 60.0, 90.0],
                                  BottomParameters.from_sediment('rock'))
        assert rock[0] < 0.1 and rock[1] < 0.1 and 2.5 < rock[2] < 3.1

    def test_loss_is_zero_at_grazing_incidence_and_frequency_free(self):
        p = BottomParameters.from_sediment('medium sand')
        assert apl_uw_bottom_loss(0.0, p) == pytest.approx(0.0, abs=1e-9)
        # no frequency argument at all: the model has none (IV.B.1)
        assert apl_uw_bottom_loss(20.0, p) == apl_uw_bottom_loss(20.0, p)


class TestBottomParameters:

    def test_table_2_rows_are_the_grain_size_relations(self):
        """Medium sand (Mz = 1.5) prints rho 1.845, nu 1.1782, delta 0.01624,
        sigma2 0.002, w2 0.004446; sandy gravel (Mz = -1) w2 0.012937; the
        silts w2 0.000518 and sigma2 0.001 above Mz 5.5."""
        p = BottomParameters.from_sediment('medium sand')
        assert p.density_ratio == pytest.approx(1.845, abs=5e-4)
        assert p.speed_ratio == pytest.approx(1.1782, abs=5e-5)
        assert p.loss_parameter == pytest.approx(0.01624, abs=5e-6)
        assert p.volume_parameter == 0.002
        assert p.spectral_strength == pytest.approx(0.004446, abs=1e-6)
        assert p.spectral_exponent == 3.25
        assert BottomParameters.from_sediment('sandy gravel').spectral_strength \
            == pytest.approx(0.012937, abs=1e-6)
        silt = BottomParameters.from_sediment('sandy mud')
        assert silt.spectral_strength == pytest.approx(0.000518, abs=1e-6)
        assert silt.volume_parameter == 0.001

    def test_names_are_normalised_and_unknown_ones_list_the_table(self):
        assert BottomParameters.from_sediment('Fine-Sand') == \
            BottomParameters.from_grain_size(2.5)
        with pytest.raises(ConfigurationError, match='medium sand'):
            BottomParameters.from_sediment('basalt')
        assert 'rough rock' in APL_UW_SEDIMENTS

    def test_geoacoustics_invert_the_speed_ratio_for_the_grain_size(self):
        """Given medium sand's own cp, rho and dB/wavelength attenuation,
        from_geoacoustics recovers Table 2's row: delta from the attenuation
        (alpha = 40 pi log10(e) delta) and w2, sigma2 from the Mz that
        inverts Eq. 3 (p. IV-12)."""
        ref = BottomParameters.from_sediment('medium sand')
        alpha_dB_lambda = ref.loss_parameter * 40.0 * np.pi / np.log(10.0)
        p = BottomParameters.from_geoacoustics(
            sound_speed=ref.speed_ratio * 1500.0, density=ref.density_ratio * 1.027,
            attenuation_dB_per_wavelength=alpha_dB_lambda,
            water_sound_speed=1500.0, water_density=1.027)
        assert p.density_ratio == pytest.approx(ref.density_ratio)
        assert p.speed_ratio == pytest.approx(ref.speed_ratio)
        assert p.loss_parameter == pytest.approx(ref.loss_parameter)
        assert p.spectral_strength == pytest.approx(ref.spectral_strength, rel=1e-3)
        assert p.volume_parameter == ref.volume_parameter

    def test_a_fetched_seabed_takes_the_grain_size_route_by_default(self):
        """A seabed built from a grain size (what every grain-size source of
        fetch_environment returns) carries grain_size_phi, and from_bottom
        reads the handbook's own relations off it whatever conversion built
        the geoacoustics — the same BottomParameters for the Hamilton and the
        APL-UW seabed of the same Mz."""
        from uacpy.data import bottom_from_grain_size
        expected = BottomParameters.from_grain_size(1.5)
        for model in ('hamilton', 'apl-uw'):
            seabed = bottom_from_grain_size(1.5, model=model,
                                            water_sound_speed=1500.0)
            assert BottomParameters.from_bottom(
                seabed, water_sound_speed=1500.0) == expected
            assert BottomParameters.from_bottom(
                seabed, water_sound_speed=1500.0, method='grain-size') == expected

    def test_the_geoacoustics_route_forms_the_ratios_against_the_given_water(self):
        """method='geoacoustics' is from_geoacoustics on the surficial cp,
        rho and dB/wavelength attenuation: the ratios follow the water values
        passed, and the seabed's own grain size still supplies sigma2 and w2.
        A boundary without a grain size takes this route under 'auto', and
        refuses 'grain-size' with the remedy."""
        from uacpy.core.environment import BoundaryProperties
        from uacpy.data import bottom_from_grain_size
        seabed = bottom_from_grain_size(1.5, model='apl-uw',
                                        water_sound_speed=1500.0)
        # The seabed was built against the package's one water (1.027).
        p = BottomParameters.from_bottom(seabed, water_sound_speed=1500.0,
                                         water_density=1.027,
                                         method='geoacoustics')
        ref = BottomParameters.from_grain_size(1.5)
        assert p.speed_ratio == pytest.approx(ref.speed_ratio)
        assert p.density_ratio == pytest.approx(ref.density_ratio)
        assert p.spectral_strength == ref.spectral_strength
        assert p.volume_parameter == ref.volume_parameter
        colder = BottomParameters.from_bottom(seabed, water_sound_speed=1450.0,
                                              water_density=1.027,
                                              method='geoacoustics')
        assert colder.speed_ratio == pytest.approx(ref.speed_ratio * 1500.0 / 1450.0)
        bare = BoundaryProperties(sound_speed=1700.0, density=1.9, attenuation=0.6)
        auto = BottomParameters.from_bottom(bare, water_sound_speed=1500.0,
                                            water_density=1.0)
        assert auto == BottomParameters.from_geoacoustics(
            sound_speed=1700.0, density=1.9, attenuation_dB_per_wavelength=0.6,
            water_sound_speed=1500.0, water_density=1.0)
        with pytest.raises(ConfigurationError, match="method='geoacoustics'"):
            BottomParameters.from_bottom(bare, water_sound_speed=1500.0,
                                         method='grain-size')

    def test_layered_and_range_dependent_seabeds_contribute_their_surface(self):
        """A SeabedColumn hands over its top layer, a Bottom the nearest
        column at ``range``; a vacuum/rigid seabed has nothing to
        parameterise and says so."""
        from uacpy.core.boundary import BoundaryProperties, SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
        from uacpy.data import bottom_from_grain_size
        sand = SedimentLayer(thickness=5.0, sound_speed=1700.0, density=1.9,
                             attenuation=0.6)
        rock = BoundaryProperties(sound_speed=3000.0, density=2.4, attenuation=0.1)
        column = SeabedColumn(layers=[sand], halfspace=rock)
        top = BottomParameters.from_bottom(column, water_sound_speed=1500.0,
                                           water_density=1.0)
        assert top.speed_ratio == pytest.approx(1700.0 / 1500.0)
        bottom = Bottom.from_columns(
            [SeabedColumn.from_halfspace(
                bottom_from_grain_size(1.5, water_sound_speed=1500.0)),
             SeabedColumn.from_halfspace(
                bottom_from_grain_size(6.0, water_sound_speed=1500.0))],
            ranges=[0.0, 1000.0])
        near = BottomParameters.from_bottom(bottom, water_sound_speed=1500.0)
        far = BottomParameters.from_bottom(bottom, water_sound_speed=1500.0,
                                           range=900.0)
        assert near == BottomParameters.from_grain_size(1.5)
        assert far == BottomParameters.from_grain_size(6.0)
        with pytest.raises(ConfigurationError, match="'vacuum'"):
            BottomParameters.from_bottom(BoundaryProperties(),
                                         water_sound_speed=1500.0)
        with pytest.raises(ConfigurationError, match='expected a BoundaryProperties'):
            BottomParameters.from_bottom('medium sand', water_sound_speed=1500.0)

    def test_from_environment_reads_the_water_at_the_seafloor(self):
        """The ratios are formed against the sound speed the environment has
        at the seafloor under ``range`` and against env.water_density."""
        from uacpy.core.environment import BoundaryProperties, Environment
        seabed = BoundaryProperties(sound_speed=1700.0, density=1.9,
                                    attenuation=0.6)
        env = Environment(bathymetry=100.0, ssp=[(0.0, 1500.0), (100.0, 1480.0)],
                          bottom=seabed, water_density=1.03)
        p = BottomParameters.from_environment(env)
        assert p.speed_ratio == pytest.approx(1700.0 / 1480.0)
        assert p.density_ratio == pytest.approx(1.9 / 1.03)
        assert p == BottomParameters.from_bottom(
            seabed, water_sound_speed=1480.0, water_density=1.03)

    def test_grain_size_outside_the_fit_is_clamped_with_a_warning(self):
        with pytest.warns(UserWarning, match='-1 <= Mz <= 9'):
            p = BottomParameters.from_grain_size(12.0)
        assert p == BottomParameters.from_grain_size(9.0)

    @pytest.mark.parametrize('phi, edge', [(-5.0, -1.0), (12.0, 9.0)])
    def test_the_geoacoustics_route_clamps_its_grain_size_with_the_same_warning(
            self, phi, edge):
        """``sigma2`` and ``w2`` are the two parameters the report says the
        backscattering strength is particularly sensitive to (p. IV-5), and in
        this constructor they are the only two the grain size feeds: w2 swings
        25x across the clamp. The measured rho, nu and delta are untouched."""
        kwargs = dict(sound_speed=1700.0, density=1.8,
                      attenuation_dB_per_wavelength=0.5,
                      water_sound_speed=1500.0, water_density=1.0)
        with pytest.warns(UserWarning, match='-1 <= Mz <= 9'):
            p = BottomParameters.from_geoacoustics(grain_size_phi=phi, **kwargs)
        at_edge = BottomParameters.from_geoacoustics(grain_size_phi=edge, **kwargs)
        assert p == at_edge
        assert p.loss_parameter == pytest.approx(
            0.5 * np.log(10.0) / (40.0 * np.pi))

    @pytest.mark.parametrize('phi', [float('nan'), float('inf')])
    def test_the_geoacoustics_route_refuses_a_non_finite_grain_size(self, phi):
        """min/max propagate a NaN, and both grain-size branches then take
        their else arm on a false comparison, so a NaN would select the
        fine-sediment sigma2 and w2 — a finite, plausible pair built out of
        nothing. The sibling constructor refuses the same input."""
        with pytest.raises(ConfigurationError, match='must be finite'):
            BottomParameters.from_geoacoustics(
                sound_speed=1700.0, density=1.8,
                attenuation_dB_per_wavelength=0.5, water_sound_speed=1500.0,
                water_density=1.0, grain_size_phi=phi)

    def test_the_two_routes_differ_in_delta_by_the_reports_c1(self):
        """from_grain_size builds delta through Eq. 4 at the c1 the report
        used; from_geoacoustics builds it from the attenuation with no c1. On
        the same seabed rho and nu agree exactly and delta differs by that
        ratio — the factor from_geoacoustics documents."""
        from uacpy.core.sediment import grain_size_to_geoacoustics
        from uacpy.sonar.bottom_scattering import TABLE_WATER_SOUND_SPEED
        water = 1500.0
        for phi in (0.5, 2.5, 5.0, 8.0):
            geo = grain_size_to_geoacoustics(phi, model='apl-uw')
            direct = BottomParameters.from_grain_size(phi)
            via_geo = BottomParameters.from_geoacoustics(
                sound_speed=geo['sound_speed'], density=geo['density'],
                attenuation_dB_per_wavelength=geo['attenuation'],
                water_sound_speed=water, water_density=1.027,
                grain_size_phi=phi)
            assert via_geo.density_ratio == pytest.approx(direct.density_ratio)
            assert via_geo.speed_ratio == pytest.approx(direct.speed_ratio)
            assert direct.loss_parameter / via_geo.loss_parameter == \
                pytest.approx(TABLE_WATER_SOUND_SPEED / water)

    def test_values_outside_the_recommended_limits_warn(self):
        with pytest.warns(UserWarning, match='Section IV.A.8'):
            BottomParameters(density_ratio=3.5, speed_ratio=1.2, loss_parameter=0.01,
                             volume_parameter=0.002, spectral_strength=0.005)
        with pytest.raises(ConfigurationError, match='between 2 and 4'):
            BottomParameters(density_ratio=1.8, speed_ratio=1.2, loss_parameter=0.01,
                             volume_parameter=0.002, spectral_strength=0.005,
                             spectral_exponent=4.0)


class TestSurfaceBackscatter:

    def test_levels_match_the_reports_model_data_figures(self):
        """Figure 3 (FLIP77, 15 kHz, 3.5 m/s): about -47 to -45 dB over
        10-30 deg; Figure 7 (SAXON-FPN, 70 kHz, 8 m/s): about -24 dB at 30 and
        -19 dB at 60 deg; Figure 8 (NOREX85, 18 kHz, 17 m/s): about -27, -21
        and -17 dB at 15, 40 and 60 deg. Read off the plots to +-2 dB."""
        flip = apl_uw_surface_backscatter(frequency=15e3,
                                          grazing_deg=[10.0, 20.0, 30.0], wind_speed_kn=ms_to_knots(3.5))
        assert np.all((-50.0 < flip) & (flip < -43.0))
        saxon = apl_uw_surface_backscatter(frequency=70e3,
                                           grazing_deg=[30.0, 60.0], wind_speed_kn=ms_to_knots(8.0))
        assert saxon[0] == pytest.approx(-24.0, abs=2.0)
        assert saxon[1] == pytest.approx(-19.0, abs=2.0)
        norex = apl_uw_surface_backscatter(frequency=18e3,
                                           grazing_deg=[15.0, 40.0, 60.0], wind_speed_kn=ms_to_knots(17.0))
        assert np.abs(norex - np.array([-27.0, -21.0, -17.0])).max() < 2.5

    def test_the_bubble_extinction_reads_the_grazing_angle(self, monkeypatch):
        """Eq. 16's two-way bubble loss enters through
        ``bubble_surface_loss(..., grazing_deg=theta)`` at the grazing angle
        the backscatter was asked for. Reading it from the normal instead
        moved the 60 kHz, 15 m/s curve by +0.04, -7.42 and -24.24 dB at 20,
        75 and 89 deg — invisible at shallow grazing — so the angle handed
        over is pinned directly."""
        import uacpy.sonar.scattering as scattering
        seen = []
        real = scattering.bubble_surface_loss

        def spy(wind_speed_kn, frequency, grazing_deg):
            seen.append(np.array(grazing_deg, dtype=float, copy=True))
            return real(wind_speed_kn, frequency, grazing_deg=grazing_deg)

        monkeypatch.setattr(scattering, 'bubble_surface_loss', spy)
        grazing = np.array([20.0, 75.0, 89.0])
        apl_uw_surface_backscatter(frequency=60e3, grazing_deg=grazing,
                                   wind_speed_kn=ms_to_knots(15.0))
        assert seen
        assert any(np.allclose(g, grazing) for g in seen)

    def test_bubbles_saturate_and_facets_dominate_near_vertical(self):
        """II.B.3: the curves stop moving with wind above about 8 m/s;
        Figure 2: every curve ends between about +3 and +8 dB at 90 deg."""
        mid = [float(apl_uw_surface_backscatter(
            frequency=25e3, grazing_deg=30.0, wind_speed_kn=ms_to_knots(u)))
            for u in (3, 5, 8, 10, 15)]
        assert mid[0] < mid[1] < mid[2]
        assert abs(mid[3] - mid[4]) < 5.0
        top = [float(apl_uw_surface_backscatter(
            frequency=25e3, grazing_deg=90.0, wind_speed_kn=ms_to_knots(u)))
            for u in (3, 10, 15)]
        assert all(2.0 < v < 9.0 for v in top)
        assert top[0] > top[2], "a rougher sea spreads the specular peak"

    def test_below_half_a_degree_is_a_linear_extrapolation(self):
        s = apl_uw_surface_backscatter(frequency=25e3,
                                       grazing_deg=[0.0, 0.25, 0.5, 1.0], wind_speed_kn=ms_to_knots(8.0))
        assert np.all(np.isfinite(s))
        assert s[1] == pytest.approx(0.5 * (s[0] + s[2]))
        assert s[2] - s[0] == pytest.approx(s[3] - s[2])

    def test_out_of_band_frequency_warns_and_bad_inputs_raise(self):
        with pytest.warns(UserWarning, match='12-70 kHz'):
            apl_uw_surface_backscatter(frequency=5e3,
                                       grazing_deg=20.0, wind_speed_kn=ms_to_knots(8.0))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            # a calm sea is valid
            apl_uw_surface_backscatter(frequency=25e3,
                                       grazing_deg=20.0, wind_speed_kn=0.0)
        with pytest.raises(ConfigurationError, match='wind_speed_kn'):
            apl_uw_surface_backscatter(frequency=25e3,
                                       grazing_deg=20.0, wind_speed_kn=-1.0)
        with pytest.raises(ConfigurationError, match='0-90'):
            apl_uw_surface_backscatter(frequency=25e3,
                                       grazing_deg=-5.0, wind_speed_kn=ms_to_knots(8.0))

    def test_wind_above_the_strongest_compared_with_data_warns(self):
        """NOREX85's 17 m/s (Figure 8) is the strongest wind the report
        compared the model with: 17 m/s is silent, 17.5 m/s warns."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            apl_uw_surface_backscatter(frequency=25e3, grazing_deg=20.0,
                                       wind_speed_kn=ms_to_knots(17.0))
        with pytest.warns(UserWarning, match='NOREX85'):
            apl_uw_surface_backscatter(frequency=25e3, grazing_deg=20.0,
                                       wind_speed_kn=ms_to_knots(17.5))

    def test_every_argument_is_keyword_only(self):
        """The two surface models took (wind, frequency) in opposite
        positional orders; a positional call is refused by both."""
        with pytest.raises(TypeError,
                           match='takes 0 positional arguments but'):
            apl_uw_surface_backscatter(20.0, 25e3, 8.0)
        with pytest.raises(TypeError,
                           match='takes 0 positional arguments but'):
            chapman_harris_surface(20.0, 10.0, 1000.0)

    @pytest.mark.parametrize('wind_mps', [3.0, 8.0, 10.0, 15.0])
    def test_facet_transition_angle_is_the_lobes_15_dB_point(self, wind_mps):
        """TR 9407 p. II-9: theta_f is the smallest grazing angle at which
        sigma_f (Eq. 11) is 15 dB below its value at 90 deg."""
        from uacpy.sonar.scattering import (
            _facet_cross_section, _facet_slope_sq, _facet_transition_deg)
        s2 = _facet_slope_sq(wind_mps)
        theta_f = _facet_transition_deg(s2)
        peak = _facet_cross_section(np.pi / 2.0, s2)
        drop = 10.0 * np.log10(
            peak / _facet_cross_section(np.deg2rad(theta_f), s2))
        assert drop == pytest.approx(15.0, abs=1e-6)


class TestTheRoughnessCriteriaRefuseANonFiniteSoundSpeed:
    """``k = 2*pi*f/c``: an infinite sound speed gives ``k = 0``, so
    ``rayleigh_parameter`` answered P = 0 (a mirror) and
    ``perturbative_grazing_limit`` 90 deg (perturbative at every angle) — the
    most reassuring answers each can give — and a NaN one answered NaN. Both
    sides of the rule: every non-finite or non-positive speed is refused, the
    smallest positive one is answered."""

    @pytest.mark.parametrize('c', [INF, NAN, 0.0, -1500.0],
                             ids=['inf', 'nan', 'zero', 'negative'])
    def test_a_non_finite_or_non_positive_speed_is_refused(self, c):
        from uacpy.sonar import perturbative_grazing_limit, rayleigh_parameter
        with pytest.raises(ConfigurationError,
                           match='rayleigh_parameter: sound_speed must be > 0 '
                                 'm/s and finite'):
            rayleigh_parameter(250.0, 0.5, 10.0, c)
        with pytest.raises(ConfigurationError,
                           match='perturbative_grazing_limit: sound_speed must '
                                 'be > 0 m/s and finite'):
            perturbative_grazing_limit(250.0, 0.5, c)

    @pytest.mark.parametrize('c', [1e-6, 1500.0], ids=['tiny', 'nominal'])
    def test_a_positive_finite_speed_is_answered(self, c):
        from uacpy.sonar import perturbative_grazing_limit, rayleigh_parameter
        assert np.isfinite(rayleigh_parameter(250.0, 0.5, 10.0, c))
        assert 0.0 <= perturbative_grazing_limit(250.0, 0.5, c) <= 90.0


class TestRayleighRoughnessParameter:
    """P = 2*k*sigma*sin(theta): the number that says whether a rough-interface
    perturbation treatment applies (JKPS Sect. 1.7)."""

    def test_matches_hand_evaluation_of_the_definition(self):
        from uacpy.sonar import rayleigh_parameter
        f, sigma, theta, c = 250.0, 0.5, 30.0, 1500.0
        k = 2 * np.pi * f / c
        expected = 2 * k * sigma * np.sin(np.deg2rad(theta))
        assert rayleigh_parameter(f, sigma, theta, c) == pytest.approx(expected)

    def test_grazing_convention_is_sine_not_cosine(self):
        # JKPS writes 2*k*sigma*sin(theta) for GRAZING theta; Brekhovskikh &
        # Lysanov Sect. 9.1 write 2*k*sigma*cos(theta_0) against INCIDENCE.
        # Same number, so the two must agree on complementary angles, and a
        # cosine slip here would be silent at 45 deg only.
        from uacpy.sonar import rayleigh_parameter
        grazing = 20.0
        p = rayleigh_parameter(250.0, 0.5, grazing)
        k = 2 * np.pi * 250.0 / 1500.0
        assert p == pytest.approx(2 * k * 0.5 * np.cos(np.deg2rad(90 - grazing)))
        # Largest at normal incidence, vanishing along the interface.
        assert rayleigh_parameter(250.0, 0.5, 90.0) > p
        assert rayleigh_parameter(250.0, 0.5, 0.0) == pytest.approx(0.0)

    def test_scales_with_frequency_and_roughness(self):
        from uacpy.sonar import rayleigh_parameter
        base = rayleigh_parameter(100.0, 0.5, 10.0)
        assert rayleigh_parameter(200.0, 0.5, 10.0) == pytest.approx(2 * base)
        assert rayleigh_parameter(100.0, 1.0, 10.0) == pytest.approx(2 * base)

    def test_reproduces_the_smooth_to_scattering_swing_of_one_sea_state(self):
        # A 0.5 m RMS sea at 10 deg grazing is a mirror at 100 Hz and a
        # scatterer at 10 kHz -- the same water, two decades apart.
        from uacpy.sonar import rayleigh_parameter
        assert rayleigh_parameter(100.0, 0.5, 10.0) == pytest.approx(0.073, abs=0.001)
        assert rayleigh_parameter(10_000.0, 0.5, 10.0) == pytest.approx(7.27, abs=0.01)

    def test_coherent_factor_is_the_jkps_exponential(self):
        from uacpy.sonar import coherent_reflection_factor, rayleigh_parameter
        p = rayleigh_parameter(250.0, 0.5, 45.0)
        assert coherent_reflection_factor(250.0, 0.5, 45.0) == pytest.approx(
            np.exp(-0.5 * p ** 2))

    def test_coherent_factor_is_unity_on_a_smooth_interface(self):
        from uacpy.sonar import coherent_reflection_factor
        assert coherent_reflection_factor(250.0, 0.0, 45.0) == pytest.approx(1.0)

    def test_grazing_limit_inverts_the_parameter_at_p_equals_one(self):
        from uacpy.sonar import perturbative_grazing_limit, rayleigh_parameter
        limit = float(perturbative_grazing_limit(250.0, 0.5))
        assert limit < 90.0
        assert rayleigh_parameter(250.0, 0.5, limit) == pytest.approx(1.0)

    def test_grazing_limit_saturates_at_ninety_when_always_perturbative(self):
        from uacpy.sonar import perturbative_grazing_limit, rayleigh_parameter
        # 2*k*sigma <= 1, so P <= 1 even at normal incidence.
        assert float(perturbative_grazing_limit(250.0, 0.05)) == pytest.approx(90.0)
        assert rayleigh_parameter(250.0, 0.05, 90.0) < 1.0

    def test_rejects_a_negative_rms_height(self):
        from uacpy.sonar import rayleigh_parameter
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='cannot be negative'):
            rayleigh_parameter(250.0, -0.5, 10.0)

    def test_warning_fires_only_past_the_limit_and_names_the_angle(self):
        from uacpy.sonar.scattering import warn_if_roughness_is_not_perturbative
        with recorded_warnings() as w:
            warn_if_roughness_is_not_perturbative('T', 250.0, 0.05)
        assert not [x for x in w if 'Rayleigh' in str(x.message)]

        with recorded_warnings() as w:
            warn_if_roughness_is_not_perturbative('T', 250.0, 0.5)
        hit = [x for x in w if 'Rayleigh' in str(x.message)]
        assert len(hit) == 1
        assert '72.7 deg' in str(hit[0].message)

    def test_warning_uses_the_requested_angles_when_given(self):
        from uacpy.sonar.scattering import warn_if_roughness_is_not_perturbative
        # Shallow grazing on the same interface stays inside the theory even
        # though normal incidence would not.
        with recorded_warnings() as w:
            warn_if_roughness_is_not_perturbative(
                'T', 250.0, 0.5, grazing_deg=[5.0, 10.0, 20.0])
        assert not [x for x in w if 'Rayleigh' in str(x.message)]

        with recorded_warnings() as w:
            warn_if_roughness_is_not_perturbative(
                'T', 250.0, 0.5, grazing_deg=[5.0, 85.0])
        assert len([x for x in w if 'Rayleigh' in str(x.message)]) == 1

    @pytest.mark.parametrize('rms, grazing', [
        (0.05, None), (0.5, None), (0.5, [5.0, 10.0, 20.0]),
        (0.5, [5.0, 85.0])])
    def test_the_notice_is_the_text_the_warning_says(self, rms, grazing):
        """``non_perturbative_roughness_notice`` (what a stage-3 resolver
        records) is the warning's text past the limit and ``None`` inside
        it, and says nothing itself."""
        from uacpy.sonar.scattering import (
            non_perturbative_roughness_notice,
            warn_if_roughness_is_not_perturbative)
        with recorded_warnings() as w:
            notice = non_perturbative_roughness_notice(
                'T', 250.0, rms, grazing_deg=grazing)
        assert w == []
        with recorded_warnings() as w:
            warn_if_roughness_is_not_perturbative(
                'T', 250.0, rms, grazing_deg=grazing)
        assert [str(x.message) for x in w] == (
            [] if notice is None else [notice])
        assert (notice is None) == (rms == 0.05 or grazing == [5.0, 10.0,
                                                               20.0])


class TestDetectionThresholdIsRequired:
    """``detection_threshold_dB`` has no default on any budget: omitting it
    raises instead of reporting the ``SNR = 0 dB`` boundary as an operating
    point."""

    @staticmethod
    def _tl_field():
        from uacpy.core.results import Field
        return Field(data=np.full((3, 4), 60.0),
                     coords={'depth': np.linspace(10.0, 50.0, 3),
                             'range': np.linspace(100.0, 4000.0, 4)})

    @pytest.mark.parametrize('call', [
        lambda: sonar.passive_signal_excess(160.0, 70.0, 60.0),
        lambda: sonar.active_signal_excess(220.0, 70.0, 10.0,
                                           noise_level_dB=60.0),
        lambda: sonar.figure_of_merit(160.0, 60.0),
    ], ids=['passive', 'active', 'figure_of_merit'])
    def test_a_scalar_budget_without_it_raises(self, call):
        with pytest.raises(TypeError, match='detection_threshold_dB'):
            call()

    @pytest.mark.parametrize('mode', ['passive', 'active'])
    def test_a_field_budget_without_it_raises(self, mode):
        extra = {} if mode == 'passive' else {'target_strength_dB': 10.0}
        call = getattr(sonar, f'{mode}_signal_excess_field')
        with pytest.raises(TypeError, match='detection_threshold_dB'):
            call(self._tl_field(), source_level_dB=160.0,
                 noise_level_dB=60.0, **extra)

    def test_it_is_keyword_only(self):
        """A DT passed in the fifth positional slot of
        ``passive_signal_excess`` is refused; by keyword it is subtracted."""
        with pytest.raises(TypeError,
                           match='positional arguments but 5 were given'):
            sonar.passive_signal_excess(160.0, 70.0, 60.0, 0.0, 10.0)
        assert sonar.passive_signal_excess(
            160.0, 70.0, 60.0, 0.0, detection_threshold_dB=10.0) == 20.0


class TestDetectionRangeField:
    """``detection_range_from_field`` is ``detection_range`` on a 1-D ``['range']``
    signal-excess Field: same numbers, same modes, same beyond-the-grid inf."""

    @staticmethod
    def _se_map():
        from uacpy.core.results import Field
        r = np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0])
        tl = np.array([[40.0, 50.0, 61.0, 50.0, 62.0],
                       [40.0, 45.0, 50.0, 55.0, 59.0]])
        field = Field(data=tl, coords={'depth': np.array([10.0, 20.0]),
                                       'range': r})
        return sonar.passive_signal_excess_field(
            field, source_level_dB=120.0, noise_level_dB=60.0,
            detection_threshold_dB=0.0)

    @pytest.mark.parametrize('crossing', ['outermost', 'first'])
    def test_it_equals_detection_range_on_the_cut(self, crossing):
        se = self._se_map()
        cut = se.at(depth=10.0)
        assert sonar.detection_range_from_field(cut, crossing=crossing) == \
            pytest.approx(sonar.detection_range(cut.coords['range'],
                                                signal_excess_dB=cut.data, crossing=crossing))
        # SE = [20, 10, -1, 10, -2] dB on 0-4 km.
        assert sonar.detection_range_from_field(cut) == \
            pytest.approx(3000.0 + 1000.0 * 10.0 / 12.0)
        assert sonar.detection_range_from_field(cut, crossing='first') == \
            pytest.approx(1000.0 + 1000.0 * 10.0 / 11.0)

    def test_a_cut_positive_at_its_edge_is_inf(self):
        cut = self._se_map().at(depth=20.0)
        assert sonar.detection_range_from_field(cut) == np.inf

    def test_a_two_dimensional_map_is_refused(self):
        with pytest.raises(ConfigurationError, match="1-D \\['range'\\]"):
            sonar.detection_range_from_field(self._se_map())

    def test_a_tl_field_is_refused(self):
        from uacpy.core.results import Field
        tl = Field(data=np.array([40.0, 70.0]),
                   coords={'range': np.array([100.0, 200.0])})
        with pytest.raises(ConfigurationError, match='not signal excess'):
            sonar.detection_range_from_field(tl)


def _budget_tl_field():
    from uacpy.core.results import Field
    depths = np.linspace(0.0, 100.0, 5)
    ranges = np.linspace(1000.0, 9000.0, 7)
    tl = 20.0 * np.log10(ranges)[None, :] + 0.1 * depths[:, None]
    return Field(data=tl, coords={'depth': depths, 'range': ranges},
                 model='Bellhop', frequencies=2000.0), tl


_RL = np.linspace(90.0, 70.0, 7)
_AG = np.linspace(6.0, 14.0, 35).reshape(5, 7)
_BUDGETS = {
    'passive': dict(mode='passive', source_level_dB=140.0,
                    noise_level_dB=60.0, directivity_index_dB=15.0,
                    detection_threshold_dB=3.0, processing_loss_dB=1.5),
    'passive_grid': dict(mode='passive', source_level_dB=140.0,
                         noise_level_dB=60.0, array_gain_dB=_AG,
                         detection_threshold_dB=-2.0),
    'active_noise': dict(mode='active', source_level_dB=220.0,
                         target_strength_dB=10.0, noise_level_dB=60.0,
                         directivity_index_dB=15.0,
                         detection_threshold_dB=3.0),
    'active_both': dict(mode='active', source_level_dB=220.0,
                        target_strength_dB=-5.0, noise_level_dB=60.0,
                        reverberation_level_dB=_RL, array_gain_dB=_AG,
                        detection_threshold_dB=0.0, processing_loss_dB=2.0),
}


class TestSonarBudgetCarriesOneBudget:
    """One budget's terms as one value: the ``*_field`` functions build it,
    evaluate it and record ``to_dict()``; ``from_dict`` of that record
    rebuilds a budget that reproduces the map exactly."""

    @pytest.mark.parametrize('name', sorted(_BUDGETS))
    def test_the_field_function_is_the_budget_over_the_field(self, name):
        field, _ = _budget_tl_field()
        terms = dict(_BUDGETS[name])
        mode = terms.pop('mode')
        se = getattr(sonar, f'{mode}_signal_excess_field')(field, **terms)
        budget = sonar.SonarBudget(mode, **terms)
        assert np.array_equal(budget.signal_excess_field(field).data, se.data)
        assert se.sonar_budget == budget.to_dict()

    @pytest.mark.parametrize('name', sorted(_BUDGETS))
    def test_the_recorded_dict_rebuilds_the_map_bit_exactly(self, name):
        field, _ = _budget_tl_field()
        se = sonar.SonarBudget(**_BUDGETS[name]).signal_excess_field(field)
        rebuilt = sonar.SonarBudget.from_dict(se.sonar_budget)
        assert rebuilt == sonar.SonarBudget(**_BUDGETS[name])
        assert np.array_equal(rebuilt.signal_excess_field(field).data,
                              se.data)

    def test_signal_excess_is_the_array_function(self):
        _, tl = _budget_tl_field()
        passive = sonar.SonarBudget(**_BUDGETS['passive'])
        assert np.array_equal(
            passive.signal_excess(tl),
            sonar.passive_signal_excess(
                140.0, tl, 60.0, 15.0, detection_threshold_dB=3.0,
                processing_loss_dB=1.5))
        active = sonar.SonarBudget(**_BUDGETS['active_both'])
        assert np.array_equal(
            active.signal_excess(tl),
            sonar.active_signal_excess(
                220.0, tl, -5.0, noise_level_dB=60.0,
                reverberation_level_dB=_RL, array_gain_dB=_AG,
                detection_threshold_dB=0.0, processing_loss_dB=2.0))

    def test_a_per_range_level_follows_range_axis(self):
        _, tl = _budget_tl_field()
        budget = sonar.SonarBudget('active', source_level_dB=220.0,
                                   target_strength_dB=10.0,
                                   reverberation_level_dB=_RL,
                                   detection_threshold_dB=0.0)
        assert np.array_equal(budget.signal_excess(tl.T, range_axis=0),
                              budget.signal_excess(tl).T)
        with pytest.raises(ConfigurationError, match='range_axis=2'):
            budget.signal_excess(tl, range_axis=2)
        # -2 is the first axis of a 2-D grid; -3 is outside it.
        assert np.array_equal(budget.signal_excess(tl.T, range_axis=-2),
                              budget.signal_excess(tl).T)
        with pytest.raises(ConfigurationError, match='range_axis=-3'):
            budget.signal_excess(tl, range_axis=-3)

    def test_the_passive_figure_of_merit_is_the_function(self):
        budget = sonar.SonarBudget(**_BUDGETS['passive'])
        assert budget.figure_of_merit() == sonar.figure_of_merit(
            140.0, 60.0, 15.0, detection_threshold_dB=3.0,
            processing_loss_dB=1.5)
        # SE is zero at a one-way TL equal to the figure of merit.
        assert float(budget.signal_excess(budget.figure_of_merit())) == 0.0

    def test_the_active_figure_of_merit_is_the_two_way_loss_at_zero_excess(
            self):
        budget = sonar.SonarBudget(**_BUDGETS['active_noise'])
        # SL + TS - (NL - DI) - DT
        assert float(budget.figure_of_merit()) == pytest.approx(
            220.0 + 10.0 - 45.0 - 3.0, abs=1e-12)
        two_way = float(budget.figure_of_merit())
        assert float(budget.signal_excess(two_way / 2.0)) == \
            pytest.approx(0.0, abs=1e-12)

    def test_array_terms_are_read_only_copies(self):
        rl = _RL.copy()
        budget = sonar.SonarBudget('active', source_level_dB=220.0,
                                   target_strength_dB=10.0,
                                   reverberation_level_dB=rl,
                                   detection_threshold_dB=0.0)
        rl[0] = 0.0
        assert budget.reverberation_level_dB[0] == 90.0
        with pytest.raises(ValueError, match='read-only'):
            budget.reverberation_level_dB[0] = 1.0

    def test_a_pickled_budget_keeps_its_terms_read_only(self):
        import pickle
        budget = sonar.SonarBudget(**_BUDGETS['active_both'])
        back = pickle.loads(pickle.dumps(budget))
        assert back == budget
        assert not back.array_gain_dB.flags.writeable
        assert not back.reverberation_level_dB.flags.writeable

    def test_equal_budgets_compare_and_hash_equal(self):
        a = sonar.SonarBudget(**_BUDGETS['passive_grid'])
        b = sonar.SonarBudget(**_BUDGETS['passive_grid'])
        assert a == b and hash(a) == hash(b)
        assert a != sonar.SonarBudget(**_BUDGETS['passive'])

    def test_the_summary_names_each_supplied_term(self):
        budget = sonar.SonarBudget(**_BUDGETS['passive'])
        assert budget.summary() == (
            'passive: SL 140 dB, NL 60 dB, DI 15 dB, DT 3 dB, L_sp 1.5 dB')


class TestSonarBudgetRefusesAnIncoherentBudget:

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ConfigurationError, match="SonarBudget: mode"):
            sonar.SonarBudget('bistatic', source_level_dB=1.0,
                              noise_level_dB=1.0, detection_threshold_dB=0.0)

    @pytest.mark.parametrize('term', ['target_strength_dB',
                                      'reverberation_level_dB'])
    def test_a_passive_budget_refuses_an_echo_term(self, term):
        with pytest.raises(ConfigurationError,
                           match=f'SonarBudget: {term} belong'):
            sonar.SonarBudget('passive', source_level_dB=1.0,
                              noise_level_dB=1.0, detection_threshold_dB=0.0,
                              **{term: 10.0})

    def test_a_passive_budget_needs_a_noise_level(self):
        with pytest.raises(ConfigurationError, match='needs noise_level_dB'):
            sonar.SonarBudget('passive', source_level_dB=1.0,
                              detection_threshold_dB=0.0)

    def test_an_active_budget_needs_a_target_strength(self):
        with pytest.raises(ConfigurationError,
                           match='needs target_strength_dB'):
            sonar.SonarBudget('active', source_level_dB=1.0,
                              noise_level_dB=1.0, detection_threshold_dB=0.0)

    def test_an_active_budget_needs_a_background(self):
        with pytest.raises(ConfigurationError,
                           match='SonarBudget: provide noise_level_dB'):
            sonar.SonarBudget('active', source_level_dB=1.0,
                              target_strength_dB=1.0,
                              detection_threshold_dB=0.0)
        # One background suffices: either one.
        sonar.SonarBudget('active', source_level_dB=1.0,
                          target_strength_dB=1.0, noise_level_dB=1.0,
                          detection_threshold_dB=0.0)
        sonar.SonarBudget('active', source_level_dB=1.0,
                          target_strength_dB=1.0, reverberation_level_dB=1.0,
                          detection_threshold_dB=0.0)

    def test_the_field_function_names_itself_in_a_refusal(self):
        field, _ = _budget_tl_field()
        with pytest.raises(ConfigurationError,
                           match='active_signal_excess_field: provide'):
            sonar.active_signal_excess_field(
                field, source_level_dB=220.0, target_strength_dB=1.0,
                detection_threshold_dB=0.0)

    @pytest.mark.parametrize('shape', [(), (7,)])
    def test_a_scalar_or_per_range_level_is_accepted(self, shape):
        rl = np.full(shape, 80.0)
        budget = sonar.SonarBudget('active', source_level_dB=1.0,
                                   target_strength_dB=1.0,
                                   reverberation_level_dB=rl,
                                   detection_threshold_dB=0.0)
        assert np.shape(budget.reverberation_level_dB) == shape

    @pytest.mark.parametrize('n', [6, 7, 8])
    def test_a_per_range_level_must_match_the_range_axis(self, n):
        field, _ = _budget_tl_field()
        call = lambda: sonar.active_signal_excess_field(  # noqa: E731
            field, source_level_dB=220.0, target_strength_dB=1.0,
            reverberation_level_dB=np.full(n, 80.0),
            detection_threshold_dB=0.0)
        if n == 7:
            assert call().data.shape == (5, 7)
        else:
            with pytest.raises(ConfigurationError,
                               match=f'active_signal_excess_field: '
                                     f'reverberation_level_dB length \\({n}\\)'):
                call()

    def test_a_per_range_level_needs_a_range_axis(self):
        from uacpy.core.results import Field
        field = Field(data=np.full((5, 7), 60.0),
                      coords={'depth': np.linspace(0.0, 100.0, 5),
                              'x': np.linspace(0.0, 600.0, 7)})
        with pytest.raises(ConfigurationError, match="carry a 'range' axis"):
            sonar.active_signal_excess_field(
                field, source_level_dB=220.0, target_strength_dB=1.0,
                reverberation_level_dB=np.full(7, 80.0),
                detection_threshold_dB=0.0)

    def test_a_two_dimensional_level_is_refused(self):
        with pytest.raises(ConfigurationError, match='1-D per-range'):
            sonar.SonarBudget('active', source_level_dB=1.0,
                              target_strength_dB=1.0,
                              reverberation_level_dB=np.ones((1, 7)),
                              detection_threshold_dB=0.0)

    @pytest.mark.parametrize('shape, fits', [
        ((5, 7), True), ((7,), True), ((1, 7), True), ((5, 1), True),
        ((6,), False), ((4, 7), False), ((2, 5, 7), False)])
    def test_a_gain_grid_must_broadcast_to_the_tl_grid_itself(self, shape,
                                                              fits):
        # (2, 5, 7) broadcasts WITH the grid but to a larger shape: refused.
        field, _ = _budget_tl_field()
        call = lambda: sonar.passive_signal_excess_field(  # noqa: E731
            field, source_level_dB=140.0, noise_level_dB=60.0,
            array_gain_dB=np.full(shape, 10.0), detection_threshold_dB=0.0)
        if fits:
            assert call().data.shape == (5, 7)
        else:
            with pytest.raises(ConfigurationError,
                               match='broadcasting to the TL grid'):
                call()

    def test_a_field_handed_to_signal_excess_names_the_field_method(self):
        field, _ = _budget_tl_field()
        budget = sonar.SonarBudget(**_BUDGETS['passive'])
        with pytest.raises(ConfigurationError,
                           match='Use SonarBudget.signal_excess_field'):
            budget.signal_excess(field)


class TestTransitionProbabilityIsTheArrayCurve:

    def test_zero_excess_is_one_half(self):
        assert float(sonar.transition_probability(0.0, sigma_dB=5.6)) == 0.5

    def test_it_is_the_normal_cdf_of_the_scaled_excess(self):
        se = np.array([-10.0, -1.0, 0.0, 2.5, np.nan])
        got = sonar.transition_probability(se, sigma_dB=4.0)
        assert np.array_equal(got[:4], norm.cdf(se[:4] / 4.0))
        assert np.isnan(got[4])

    def test_the_field_form_is_this_curve(self):
        field, _ = _budget_tl_field()
        se = sonar.SonarBudget(**_BUDGETS['passive']).signal_excess_field(
            field)
        pd = sonar.transition_probability_field(se, sigma_dB=5.6)
        assert np.array_equal(
            pd.data, sonar.transition_probability(se.data, sigma_dB=5.6))

    @pytest.mark.parametrize('sigma', [0.0, -1.0, np.inf, np.nan])
    def test_a_sigma_that_is_not_positive_and_finite_is_refused(self, sigma):
        with pytest.raises(ConfigurationError,
                           match='transition_probability: sigma_dB'):
            sonar.transition_probability(1.0, sigma_dB=sigma)

    def test_a_tiny_positive_sigma_is_accepted(self):
        assert float(sonar.transition_probability(1.0, sigma_dB=1e-300)) \
            == 1.0

    def test_a_field_is_refused_naming_the_field_form(self):
        field, _ = _budget_tl_field()
        with pytest.raises(ConfigurationError,
                           match='Use transition_probability_field'):
            sonar.transition_probability(field, sigma_dB=5.6)


class TestDetectionRangesAppliesDetectionRangePerSlice:

    ROWS = np.array([[10, 8, 6, 3, 1, -1, -2, 1, -3, -4, -5.0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0],
                     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1.0],
                     [5, 4, np.nan, 2, -1, np.nan, -3, -4, 2, 3, 1.0]])
    RANGES = np.linspace(0.0, 10000.0, 11)

    @pytest.mark.parametrize('crossing', ['outermost', 'first'])
    def test_each_row_is_detection_range(self, crossing):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            got = sonar.detection_ranges(self.RANGES,
                                         signal_excess_dB=self.ROWS,
                                         crossing=crossing)
            want = [sonar.detection_range(self.RANGES, signal_excess_dB=row,
                                          crossing=crossing)
                    for row in self.ROWS]
        np.testing.assert_array_equal(got, want)

    def test_the_crossing_keyword_is_the_one_detection_range_takes(self):
        """One word per idea: the three detection-range functions all name
        the choice ``crossing``; ``mode=`` is no keyword of any."""
        import inspect
        for fn in (sonar.detection_range, sonar.detection_ranges,
                   sonar.detection_ranges_by_depth):
            params = inspect.signature(fn).parameters
            assert 'crossing' in params and 'mode' not in params, fn
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            first = sonar.detection_ranges(self.RANGES,
                                           signal_excess_dB=self.ROWS,
                                           crossing='first')
            outer = sonar.detection_ranges(self.RANGES,
                                           signal_excess_dB=self.ROWS)
        assert not np.array_equal(first, outer, equal_nan=True)

    def test_the_range_axis_may_be_any_axis(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            last = sonar.detection_ranges(self.RANGES,
                                          signal_excess_dB=self.ROWS)
            first = sonar.detection_ranges(self.RANGES,
                                           signal_excess_dB=self.ROWS.T,
                                           axis=0)
            stacked = sonar.detection_ranges(
                self.RANGES, signal_excess_dB=np.stack([self.ROWS] * 3),
                axis=2)
        np.testing.assert_array_equal(first, last)
        assert stacked.shape == (3, 4)
        np.testing.assert_array_equal(stacked[1], last)

    def test_a_range_axis_of_another_length_is_refused(self):
        with pytest.raises(ConfigurationError, match='11 samples along '
                                                     'axis 0, ranges_m has 4'):
            sonar.detection_ranges(self.RANGES[:4],
                                   signal_excess_dB=self.ROWS.T, axis=0)

    def test_a_scalar_excess_is_refused(self):
        with pytest.raises(ConfigurationError, match='range axis'):
            sonar.detection_ranges(self.RANGES, signal_excess_dB=1.0)

    def test_the_depth_map_form_is_this_function(self):
        from uacpy.core.results import Field
        se = Field(data=self.ROWS, coords={'depth': np.arange(4.0),
                                           'range': self.RANGES},
                   kind='signal_excess', unit='dB')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            depths, ranges = sonar.detection_ranges_by_depth(se)
            want = sonar.detection_ranges(self.RANGES,
                                          signal_excess_dB=self.ROWS, axis=1)
        np.testing.assert_array_equal(ranges, want)
        np.testing.assert_array_equal(depths, np.arange(4.0))


@pytest.mark.parametrize('name', ['ts_sphere', 'ts_convex', 'ts_ellipsoid',
                                  'ts_cylinder', 'ts_plate'])
def test_target_strength_names_frequency_as_the_rest_of_sonar_does(name):
    """The five ts_* functions spelled it ``frequency_hz`` beside
    ``chapman_harris_surface(frequency, …)`` and 36 other ``frequency``
    parameters; the unit is the package's default (Hz), so no suffix."""
    import inspect
    params = inspect.signature(getattr(sonar, name)).parameters
    assert 'frequency' in params and 'frequency_hz' not in params
    assert params['frequency'].kind is inspect.Parameter.KEYWORD_ONLY
