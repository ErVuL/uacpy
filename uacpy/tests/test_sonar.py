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
import pytest
from scipy.stats import norm

from uacpy import sonar
from uacpy.core.exceptions import ConfigurationError
from uacpy.sonar import target_strength as ts
from uacpy.sonar.detection import (detection_threshold_energy,
                                   probability_of_detection, roc_curve)
from uacpy.sonar.reverberation import (boundary_reverberation,
                                       volume_reverberation)
from uacpy.sonar.scattering import (chapman_harris_surface,
                                    column_scattering_strength)

NAN = float('nan')
INF = float('inf')


class TestScattering:
    def test_lambert_at_grazing_90_equals_mu(self):
        # sin(90)=1 -> 20*log10(1)=0, so S_b = mu_db.
        assert sonar.lambert_bottom(90.0, mu_db=-27.0) == pytest.approx(-27.0)

    def test_lambert_mu_db_constant(self):
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
        # Independent hand value (TL=60, cell=7500 m², 10log10(7500)=38.751):
        # 220 - 120 - 40 + 38.751 = 98.751 dB — anchors the formula, not just
        # re-derives it.
        assert rl[0] == pytest.approx(98.751, abs=0.01)

    def test_tl_db_callable_matches_precomputed(self):
        # tl_db accepts a callable r -> TL(r); the equation evaluates it on
        # the range grid, matching the same TL passed as a precomputed array.
        r = np.array([500.0, 1000.0, 2000.0, 4000.0])
        tl_fn = lambda rr: 15.0 * np.log10(rr)  # noqa: E731
        b_call = sonar.boundary_reverberation(
            r, 200.0, -27.0, pulse_length_s=0.01,
            horizontal_beamwidth_rad=0.1, tl_db=tl_fn,
        )
        b_arr = sonar.boundary_reverberation(
            r, 200.0, -27.0, pulse_length_s=0.01,
            horizontal_beamwidth_rad=0.1, tl_db=15.0 * np.log10(r),
        )
        np.testing.assert_allclose(b_call, b_arr)
        v_call = sonar.volume_reverberation(
            r, 200.0, -70.0, pulse_length_s=0.01,
            solid_angle_beamwidth_sr=0.01, tl_db=tl_fn,
        )
        v_arr = sonar.volume_reverberation(
            r, 200.0, -70.0, pulse_length_s=0.01,
            solid_angle_beamwidth_sr=0.01, tl_db=15.0 * np.log10(r),
        )
        np.testing.assert_allclose(v_call, v_arr)
        # Spherical-spreading callable reproduces the tl_db=None default.
        b_sph = sonar.boundary_reverberation(
            r, 200.0, -27.0, pulse_length_s=0.01,
            horizontal_beamwidth_rad=0.1,
            tl_db=lambda rr: 20.0 * np.log10(rr),
        )
        b_none = sonar.boundary_reverberation(
            r, 200.0, -27.0, pulse_length_s=0.01,
            horizontal_beamwidth_rad=0.1, tl_db=None,
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

    def test_bad_pulse_raises(self):
        with pytest.raises(ConfigurationError):
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
        assert sonar.detection_range(r, se) == pytest.approx(9500.0)
        # an all-NaN row is genuinely unknown -> nan, not inf
        assert np.isnan(sonar.detection_range(r, np.full(6, np.nan)))

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
        # Independent hand value: reverb (90) swamps noise-DI (45), so
        # background ≈ 90.0 dB → SE ≈ 220-120+10-90-10 = 10.0 dB.
        assert se == pytest.approx(10.0, abs=0.01)

    def test_active_requires_a_background(self):
        with pytest.raises(ConfigurationError):
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

    def test_detection_range_far_edge_recovery(self):
        # Convergence-zone shape (+, -, +): the outermost positive range wins,
        # not the inner down-crossing.
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        se = np.array([5.0, -1.0, 2.0, 3.0, 4.0])  # ends positive at r=4
        assert sonar.detection_range(r, se) == pytest.approx(4.0)

    def test_detection_range_outermost_of_multiple_crossings(self):
        r = np.array([0.0, 1.0, 2.0, 3.0])
        se = np.array([5.0, -1.0, 1.0, -1.0])  # down-crossing between r=2 and r=3
        assert sonar.detection_range(r, se) == pytest.approx(2.5)

    def test_detection_range_does_not_interpolate_across_a_no_data_hole(self):
        # The + and - samples bracket an unfilled (NaN) cell, so the crossing
        # lies somewhere the model never computed; the answer is the last
        # positive sample's range, not a sub-cell position inside the hole.
        r = np.array([0.0, 1000.0, 2000.0])
        se = np.array([5.0, np.nan, -5.0])
        assert sonar.detection_range(r, se) == pytest.approx(0.0)

    def test_detection_range_interpolates_between_adjacent_samples(self):
        r = np.array([0.0, 1000.0])
        se = np.array([5.0, -5.0])
        assert sonar.detection_range(r, se) == pytest.approx(500.0)


class TestFarEdgeRecoveryUnderTrailingNoData:
    """The far-edge bound is the last range WITH DATA, not ``ranges[-1]``.

    ``detection_range`` masks the no-data cells before it looks for the far
    edge, which is why ``docs/guide/sonar.md`` §3 tells the reader that
    ``r_det < ranges[-1]`` does not separate a lower bound from a modelled
    crossing. Pinning the warning alone leaves the return value free to
    become ``ranges[-1]``, which still warns and still falsifies the page.
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

    def test_far_edge_recovery_returns_the_last_range_with_data(self):
        r, se = self._recovery_with_trailing_no_data(3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r_det = sonar.detection_range(r, se)
        assert r[-1] == 20000.0          # the edge an r < ranges[-1] guard reads
        assert r_det == pytest.approx(17000.0)

    def test_by_depth_returns_each_rows_own_last_range_with_data(self):
        """Each row masks its own no-data cells, so "the last range with data"
        is a per-row quantity and no single number can stand for the field."""
        from uacpy.core.results import Field
        r, shallow = self._recovery_with_trailing_no_data(3)
        _, deep = self._recovery_with_trailing_no_data(8)
        field = Field(
            data=np.vstack([shallow, deep]),
            coords={'depth': np.array([10.0, 20.0]), 'range': r},
            model='Bellhop',
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            depths, ranges = sonar.detection_range_by_depth(field)
        np.testing.assert_array_equal(depths, [10.0, 20.0])
        assert ranges[0] == pytest.approx(17000.0)
        assert ranges[1] == pytest.approx(12000.0)


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
        dt = sonar.detection_threshold_energy(0.9, 0.01, 100.0, 1.0)
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
            0.5, 1e-4, bandwidth_hz=50.0, integration_time_s=10.0)
        dt_active = sonar.detection_threshold_energy(
            0.5, 1e-4, bandwidth_hz=100.0, integration_time_s=0.5)
        assert dt_passive == pytest.approx(5.0 * np.log10(d / 500.0))
        assert dt_active == pytest.approx(5.0 * np.log10(d / 50.0))
        assert dt_passive == pytest.approx(-7.7906, abs=5e-4)
        assert dt_active == pytest.approx(-2.7906, abs=5e-4)
        assert dt_active - dt_passive == pytest.approx(5.0, abs=1e-9)

    def test_detection_threshold_falls_with_time_bandwidth(self):
        # Incoherent integration lowers the required SNR: 5 dB per decade of w*t
        # (Abraham §9.2). DT must DECREASE as bandwidth or integration time grow.
        base = sonar.detection_threshold_energy(0.9, 0.01, 100.0, 1.0)
        more_bw = sonar.detection_threshold_energy(0.9, 0.01, 1000.0, 1.0)
        more_t = sonar.detection_threshold_energy(0.9, 0.01, 100.0, 10.0)
        assert more_bw < base and more_t < base
        assert more_bw == pytest.approx(base - 5.0, abs=1e-9)  # one decade of w
        assert more_t == pytest.approx(base - 5.0, abs=1e-9)   # one decade of t

    def test_bad_probability_raises(self):
        with pytest.raises(ConfigurationError):
            sonar.deflection_coefficient(1.0, 0.01)


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
            field, source_level=140.0, noise_level=60.0,
            directivity_index=15.0, detection_threshold=3.0,
        )
        expected = sonar.passive_signal_excess(
            140.0, tl, 60.0, directivity_index=15.0, detection_threshold=3.0,
        )
        np.testing.assert_allclose(se.data, expected)
        assert list(se.coords) == ['depth', 'range']
        np.testing.assert_array_equal(se.coords['range'], field.coords['range'])
        assert se.model == 'Bellhop'
        assert se.metadata['sonar_budget']['mode'] == 'passive'

    def test_complex_pressure_field_converts_via_tl(self):
        field, tl = self._tl_field(complex_data=True)
        se = sonar.passive_signal_excess_field(
            field, source_level=140.0, noise_level=60.0,
        )
        expected = 140.0 - tl - 60.0
        np.testing.assert_allclose(se.data, expected, atol=1e-9)

    def test_active_per_range_reverberation_broadcast(self):
        field, tl = self._tl_field()
        rl = np.linspace(90.0, 70.0, field.coords['range'].size)
        se = sonar.active_signal_excess_field(
            field, source_level=220.0, target_strength=10.0,
            noise_level=60.0, directivity_index=15.0,
            reverberation_level=rl, detection_threshold=3.0,
        )
        expected = sonar.active_signal_excess(
            220.0, tl, 10.0, noise_level=60.0, directivity_index=15.0,
            reverberation_level=rl[None, :], detection_threshold=3.0,
        )
        np.testing.assert_allclose(se.data, expected)
        assert se.metadata['sonar_budget']['mode'] == 'active'

    def test_sonar_budget_key_sets(self):
        # Passive budget carries exactly the six always-present terms;
        # 'array_gain' joins the set only when passed.
        field, _ = self._tl_field()
        se = sonar.passive_signal_excess_field(
            field, source_level=140.0, noise_level=60.0,
            directivity_index=15.0, detection_threshold=3.0,
        )
        assert set(se.metadata['sonar_budget']) == {
            'mode', 'source_level', 'noise_level', 'directivity_index',
            'detection_threshold', 'processing_loss_db',
        }
        se_ag = sonar.passive_signal_excess_field(
            field, source_level=140.0, noise_level=60.0, array_gain=12.0,
        )
        assert set(se_ag.metadata['sonar_budget']) == {
            'mode', 'source_level', 'noise_level', 'directivity_index',
            'detection_threshold', 'processing_loss_db', 'array_gain',
        }
        # Active budget: five always-present terms, plus 'noise_level',
        # 'reverberation_level' and 'array_gain' when supplied.
        rl = np.linspace(90.0, 70.0, field.coords['range'].size)
        se_act = sonar.active_signal_excess_field(
            field, source_level=220.0, target_strength=10.0,
            noise_level=60.0, reverberation_level=rl, array_gain=12.0,
        )
        assert set(se_act.metadata['sonar_budget']) == {
            'mode', 'source_level', 'target_strength', 'directivity_index',
            'detection_threshold', 'processing_loss_db',
            'noise_level', 'reverberation_level', 'array_gain',
        }
        se_min = sonar.active_signal_excess_field(
            field, source_level=220.0, target_strength=10.0,
            noise_level=60.0,
        )
        assert set(se_min.metadata['sonar_budget']) == {
            'mode', 'source_level', 'target_strength', 'directivity_index',
            'detection_threshold', 'processing_loss_db', 'noise_level',
        }

    def test_se_field_kind_and_unit(self):
        # The SE Field is tagged kind='signal_excess' and reports unit='dB'
        # — dB but neither pressure nor a loss, so .max() finds the best cell.
        field, _ = self._tl_field()
        se = sonar.passive_signal_excess_field(
            field, source_level=140.0, noise_level=60.0,
        )
        assert se.kind == 'signal_excess'
        assert se.unit == 'dB'
        assert se.metadata['kind'] == 'signal_excess'
        se_act = sonar.active_signal_excess_field(
            field, source_level=220.0, target_strength=10.0,
            noise_level=60.0,
        )
        assert se_act.kind == 'signal_excess'
        assert se_act.unit == 'dB'

    def test_reverberation_length_mismatch_raises(self):
        field, _ = self._tl_field()
        with pytest.raises(ConfigurationError):
            sonar.active_signal_excess_field(
                field, source_level=220.0, target_strength=10.0,
                reverberation_level=np.zeros(3),
            )

    def test_time_series_field_rejected(self):
        from uacpy.core.results import Field
        f = Field(
            data=np.zeros(16),
            coords={'time': np.linspace(0.0, 1.0, 16)},
        )
        with pytest.raises(ConfigurationError):
            sonar.passive_signal_excess_field(
                f, source_level=140.0, noise_level=60.0,
            )

    def test_non_field_rejected(self):
        with pytest.raises(ConfigurationError):
            sonar.passive_signal_excess_field(
                np.zeros((3, 3)), source_level=140.0, noise_level=60.0,
            )

    def test_plot_signal_excess_smoke(self):
        import matplotlib.pyplot as plt
        from uacpy.visualization.plots import plot_signal_excess
        field, _ = self._tl_field()
        se = sonar.passive_signal_excess_field(
            field, source_level=140.0, noise_level=60.0,
            directivity_index=15.0, detection_threshold=20.0,
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
        with pytest.raises(ConfigurationError):
            plot_signal_excess(field)
        se = sonar.passive_signal_excess_field(
            field, source_level=140.0, noise_level=60.0,
        )
        with pytest.raises(ConfigurationError):
            plot_signal_excess(se.at(depth=50.0))


class TestBudgetKnobs:
    def test_array_gain_replaces_di(self):
        bg = sonar.noise_background(60.0, array_gain=12.0)
        assert bg == pytest.approx(48.0)
        se = sonar.passive_signal_excess(140.0, 70.0, 60.0, array_gain=12.0)
        assert se == pytest.approx(140.0 - 70.0 - 48.0)

    def test_array_gain_plus_di_raises(self):
        with pytest.raises(ConfigurationError):
            sonar.noise_background(60.0, 15.0, array_gain=12.0)
        with pytest.raises(ConfigurationError):
            sonar.passive_signal_excess(
                140.0, 70.0, 60.0, directivity_index=15.0, array_gain=12.0,
            )

    def test_array_gain_alone_uses_ag(self):
        # directivity_index defaults to None ("not supplied"), so array_gain
        # alone is accepted and applied — no spurious both-supplied rejection.
        assert sonar.noise_background(60.0, array_gain=12.0) == pytest.approx(48.0)

    def test_di_array_with_zero_plus_ag_raises(self):
        # An explicit per-angle DI array containing a 0 is "supplied" — mixing
        # it with array_gain is categorically an error (no 0.0 sentinel escape).
        with pytest.raises(ConfigurationError):
            sonar.noise_background(60.0, np.array([0.0, 10.0]), array_gain=12.0)

    def test_processing_loss_subtracts(self):
        base = sonar.passive_signal_excess(140.0, 70.0, 60.0,
                                           directivity_index=15.0)
        lossy = sonar.passive_signal_excess(140.0, 70.0, 60.0,
                                            directivity_index=15.0,
                                            processing_loss_db=3.0)
        assert lossy == pytest.approx(base - 3.0)
        fom = sonar.figure_of_merit(140.0, 60.0, 15.0,
                                    processing_loss_db=3.0)
        assert fom == pytest.approx(140.0 - 45.0 - 3.0)

    def test_active_array_gain_applies_to_noise_not_reverb(self):
        # Reverb-only: AG must change nothing.
        se_rl = sonar.active_signal_excess(
            220.0, 60.0, 10.0, reverberation_level=80.0,
        )
        se_rl_ag = sonar.active_signal_excess(
            220.0, 60.0, 10.0, reverberation_level=80.0, array_gain=12.0,
        )
        assert se_rl_ag == pytest.approx(se_rl)
        # Noise-only: AG acts exactly like DI of the same value.
        se_di = sonar.active_signal_excess(
            220.0, 60.0, 10.0, noise_level=70.0, directivity_index=12.0,
        )
        se_ag = sonar.active_signal_excess(
            220.0, 60.0, 10.0, noise_level=70.0, array_gain=12.0,
        )
        assert se_ag == pytest.approx(se_di)

    def test_field_variants_thread_knobs(self):
        field, tl = TestSignalExcessField._tl_field()
        se = sonar.passive_signal_excess_field(
            field, source_level=140.0, noise_level=60.0,
            array_gain=12.0, processing_loss_db=3.0,
        )
        expected = 140.0 - tl - 48.0 - 3.0
        np.testing.assert_allclose(se.data, expected)
        assert se.metadata['sonar_budget']['array_gain'] == 12.0
        assert se.metadata['sonar_budget']['processing_loss_db'] == 3.0


class TestDetectionProbabilityField:
    @staticmethod
    def _se_field(values):
        from uacpy.core.results import Field
        values = np.asarray(values, dtype=float)
        return Field(
            data=values,
            coords={'depth': np.arange(values.shape[0], dtype=float),
                    'range': np.arange(values.shape[1], dtype=float) + 1.0},
            model='Bellhop',
        )

    def test_transition_curve_anchors(self):
        # Pd = Phi(SE/sigma): 0.5 at SE=0, Phi(±1) at SE=±sigma.
        sigma = 5.6
        se = self._se_field([[0.0, sigma, -sigma, 2 * sigma]])
        pd = sonar.probability_of_detection_field(se, sigma_db=sigma)
        expected = norm.cdf(np.array([0.0, 1.0, -1.0, 2.0]))
        np.testing.assert_allclose(pd.data[0], expected, atol=1e-12)
        assert pd.metadata['sigma_db'] == pytest.approx(sigma)
        assert list(pd.coords) == ['depth', 'range']

    def test_pd_field_kind_and_unit(self):
        # The P_D Field is tagged kind='probability_of_detection' with
        # unit='1' — a dimensionless 0-1 probability, not dB.
        se = self._se_field([[0.0, 6.0]])
        pd = sonar.probability_of_detection_field(se, sigma_db=6.0)
        assert pd.kind == 'probability_of_detection'
        assert pd.unit == '1'
        assert pd.metadata['kind'] == 'probability_of_detection'
        assert pd.metadata['unit'] == '1'

    def test_monotonic_in_se_and_bounded(self):
        se = self._se_field(np.linspace(-30, 30, 61).reshape(1, -1))
        pd = sonar.probability_of_detection_field(se, sigma_db=6.0)
        assert np.all(np.diff(pd.data[0]) > 0)
        assert pd.data.min() >= 0.0 and pd.data.max() <= 1.0

    def test_bad_sigma_raises(self):
        se = self._se_field([[0.0]])
        with pytest.raises(ConfigurationError):
            sonar.probability_of_detection_field(se, sigma_db=0.0)
        with pytest.raises(ConfigurationError):
            sonar.probability_of_detection_field(se, sigma_db=-1.0)

    def test_non_field_and_complex_rejected(self):
        with pytest.raises(ConfigurationError):
            sonar.probability_of_detection_field(
                np.zeros((2, 2)), sigma_db=6.0,
            )
        from uacpy.core.results import Field
        cplx = Field(
            data=np.zeros((1, 2), dtype=complex),
            coords={'depth': [0.0], 'range': [1.0, 2.0]},
        )
        with pytest.raises(ConfigurationError):
            sonar.probability_of_detection_field(cplx, sigma_db=6.0)

    def test_detection_range_by_depth(self):
        # Row 0 crosses zero between samples; row 1 all positive (inf);
        # row 2 all negative (nan).
        r = np.array([1000.0, 2000.0, 3000.0])
        se = self._se_field([[10.0, 0.0, -10.0],
                             [5.0, 5.0, 5.0],
                             [-5.0, -5.0, -5.0]])
        se.coords['range'] = r
        depths, dr = sonar.detection_range_by_depth(se)
        assert depths.shape == dr.shape == (3,)
        assert dr[0] == pytest.approx(2000.0)   # exact zero at 2 km
        assert np.isinf(dr[1])
        assert np.isnan(dr[2])

    def test_detection_range_by_depth_requires_canonical(self):
        from uacpy.core.results import Field
        f = Field(data=np.zeros(4), coords={'range': np.arange(4.0) + 1})
        with pytest.raises(ConfigurationError):
            sonar.detection_range_by_depth(f)

    def test_plot_detection_probability_smoke(self):
        import matplotlib.pyplot as plt
        from uacpy.visualization.plots import plot_detection_probability
        se = self._se_field(
            np.linspace(20, -20, 40).reshape(2, 20),
        )
        pd = sonar.probability_of_detection_field(se, sigma_db=5.6)
        fig, ax = plot_detection_probability(pd)
        assert ax.get_xlabel() == 'Range (km)'
        assert 'σ = 5.6 dB' in ax.get_title()
        plt.close(fig)
        with pytest.raises(ConfigurationError):
            plot_detection_probability(pd.at(depth=0.0))


class TestTargetStrength:
    def test_two_metre_sphere_is_zero_db(self):
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
        ts = sonar.ts_cylinder(1.0, 5.0, 1000.0, sound_speed=1500.0)
        assert ts == pytest.approx(9.208, abs=0.01)

    def test_cylinder_aspect_pattern(self):
        # First null at β = kL·sinθ = π, i.e. sinθ = λ/(2L)
        # (Abraham §3.4: null-to-null main lobe ≈ λ/L).
        L, f, c = 5.0, 1000.0, 1500.0
        lam = c / f
        theta_null = np.degrees(np.arcsin(lam / (2.0 * L)))
        ts_null = sonar.ts_cylinder(1.0, L, f, angle_deg=theta_null,
                                    sound_speed=c)
        # Evaluated exactly on the analytic null, sinc^2 bottoms out at the
        # float64 sin() floor: TS is -306 dB here. -60 dB is a "this is a
        # null, not a lobe" threshold with ~245 dB of headroom, so it does not
        # pin the null depth, only its location.
        assert ts_null < -60.0
        # Monotone decrease across the main lobe.
        angles = np.linspace(0.0, theta_null * 0.95, 10)
        ts = sonar.ts_cylinder(1.0, L, f, angle_deg=angles, sound_speed=c)
        assert np.all(np.diff(ts) < 0)

    def test_plate_normal_incidence(self):
        # TS = 20log10(ab/λ): 1 m × 1 m plate at λ = 1 m -> 0 dB.
        ts = sonar.ts_plate(1.0, 1.0, 1500.0, sound_speed=1500.0)
        assert ts == pytest.approx(0.0)

    def test_geometric_regime_warning(self):
        # ka = 2π·50/1500·0.5 ≈ 0.105 « 10 -> Rayleigh regime, must warn.
        with pytest.warns(UserWarning, match="geometric"):
            sonar.ts_sphere(0.5, frequency_hz=50.0)
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
            sonar.ts_cylinder(0.05, 5.0, 1000.0)
        # The plate check scale is a full dimension, so its bound is one
        # wavelength (kw > 2*pi), not ka > 1: min(0.1, 0.2) = 0.1 m at
        # lambda = 1.5 m warns, and so does a 1 m plate (0.67 lambda).
        with pytest.warns(UserWarning, match="geometric"):
            sonar.ts_plate(0.1, 0.2, 1000.0)
        with pytest.warns(UserWarning, match="geometric"):
            sonar.ts_plate(1.0, 1.0, 1000.0)
        # Above the bounds both stay silent: cylinder a = 0.5 m -> ka = 2.09;
        # a 3 m plate is 2 wavelengths across at 1 kHz.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            sonar.ts_cylinder(0.5, 5.0, 1000.0)
            sonar.ts_plate(3.0, 3.0, 1000.0)

    def test_plate_warns_in_rayleigh_regime(self):
        # A 0.2 x 0.2 m plate at lambda = 1 m is 0.2 wavelengths across —
        # deep in the Rayleigh regime, where physical optics does not apply.
        # It passes the ka > 1 bound, so only the wavelength check flags it;
        # the level itself is the plate's -28.0 dB either way, which is what
        # makes the warning the whole of the protection here.
        with pytest.warns(UserWarning, match="geometric"):
            ts = sonar.ts_plate(0.2, 0.2, 1500.0, sound_speed=1500.0)
        assert ts == pytest.approx(20.0 * np.log10(0.04))

    def test_needle_ellipsoid_warns_although_k_times_semi_axis_is_geometric(self):
        """``ts_ellipsoid`` delegates to ``ts_convex(b²/a, c²/a)``, so its
        ``ka > 10`` check runs on the principal radii of curvature at the tip
        of the ``a`` axis — not on any semi-axis. A needle ``a = 10 m``,
        ``b = c = 1 m`` at 1 kHz has ``k·a = 41.9``, comfortably geometric,
        while the radii are ``b²/a = 0.1 m`` and the check sees 0.42. The
        message carries the number so a reader who did the semi-axis
        arithmetic can see which radius was tested, and it names
        ``ts_convex`` rather than the function that was called.
        """
        with pytest.warns(UserWarning, match="geometric") as caught:
            sonar.ts_ellipsoid(10.0, 1.0, 1.0, frequency_hz=1000.0,
                               sound_speed=1500.0)
        message = str(caught[0].message)
        assert 'k·a = 0.42' in message
        assert '41.9' not in message
        assert message.startswith('ts_convex:')

    def test_oblate_ellipsoid_is_silent_although_k_times_semi_axis_is_not(self):
        """The same substitution run the other way: an oblate ``a = 1 m``,
        ``b = c = 10 m`` at 1 kHz has ``k·a = 4.19``, below the bound, while
        the radii are ``b²/a = 100 m`` and the check sees 419. Silence here
        and a warning in the needle case together say WHICH radius is
        tested; either one alone is also explained by a semi-axis check.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sonar.ts_ellipsoid(1.0, 10.0, 10.0, frequency_hz=1000.0,
                               sound_speed=1500.0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ConfigurationError):
            sonar.ts_sphere(0.0)
        with pytest.raises(ConfigurationError):
            sonar.ts_cylinder(1.0, 5.0, -100.0)
        with pytest.raises(ConfigurationError):
            sonar.ts_plate(1.0, -1.0, 1000.0)

    def test_feeds_active_signal_excess(self):
        ts = sonar.ts_cylinder(0.5, 10.0, 2000.0)
        se = sonar.active_signal_excess(
            190.0, 60.0, ts, noise_level=75.0, directivity_index=15.0,
        )
        assert np.isfinite(se)


class TestDetectionThresholdReference:
    """Pin the DT convention and the 10*log10(w) offset to Urick's form."""

    def test_five_db_per_decade_of_time_bandwidth(self):
        """Abraham 9.2.3.1: SNR_d falls 5 dB per decade of M = w*t."""
        from uacpy.sonar import detection_threshold_energy
        a = detection_threshold_energy(0.5, 1e-4, bandwidth_hz=100.0,
                                       integration_time_s=1.0)
        b = detection_threshold_energy(0.5, 1e-4, bandwidth_hz=100.0,
                                       integration_time_s=10.0)
        assert (a - b) == pytest.approx(5.0, abs=1e-9)

    def test_offset_from_urick_band_power_form_is_10log10_w(self):
        """This DT is the unitless S0/N0 ratio; Urick's d*w/t form is
        referenced to noise in a 1-Hz band. They differ by 10*log10(w)."""
        import numpy as np
        from uacpy.sonar import detection_threshold_energy, detection_index
        pd, pf, w, t = 0.5, 1e-4, 100.0, 2.0
        this = detection_threshold_energy(pd, pf, w, t)
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
    def _exact_dt_db(pd, pf, m):
        from scipy.stats import gamma
        return 10.0 * np.log10(gamma.isf(pf, m) / gamma.isf(pd, m) - 1.0)

    def test_the_exact_benchmark_recovers_its_own_operating_point(self):
        from scipy.stats import gamma
        pd, pf, m = 0.9, 1e-6, 40.0
        s = 10.0 ** (self._exact_dt_db(pd, pf, m) / 10.0)
        h = gamma.isf(pf, m)
        assert gamma.sf(h, m) == pytest.approx(pf, rel=1e-9)
        assert gamma.sf(h / (1.0 + s), m) == pytest.approx(pd, rel=1e-9)

    @pytest.mark.parametrize('pd, pf, m, expected_db', [
        (0.9, 1e-6, 1.0, -13.337),
        (0.9, 1e-6, 10.0, -3.485),
        (0.9, 1e-6, 100.0, -1.070),
        (0.99, 1e-6, 1.0, -22.879),
    ])
    def test_the_shipped_value_is_optimistic_by_the_documented_amount(
            self, pd, pf, m, expected_db):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got = detection_threshold_energy(pd, pf, m, 1.0)
        error = got - self._exact_dt_db(pd, pf, m)
        assert error < 0.0
        assert error == pytest.approx(expected_db, abs=5e-3)

    def test_the_warning_boundary_is_the_1_db_promise_itself(self):
        """The guard's threshold and the accuracy claim are one thing, so the
        two sides of it are the two sides of the promise. At pd=0.9, pf=1e-6
        the crossing sits at M = 115."""
        pd, pf = 0.9, 1e-6
        # The measurements run under an explicit filter so the boundary is
        # pinned to the code, not to whatever -W the suite is invoked with.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert abs(self._exact_dt_db(pd, pf, 114.0)
                       - detection_threshold_energy(pd, pf, 114.0, 1.0)) >= 1.0
            assert abs(self._exact_dt_db(pd, pf, 115.0)
                       - detection_threshold_energy(pd, pf, 115.0, 1.0)) < 1.0
        with pytest.warns(UserWarning, match='large-M approximation'):
            detection_threshold_energy(pd, pf, 114.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            detection_threshold_energy(pd, pf, 115.0, 1.0)

    def test_the_warning_names_the_measured_error_at_this_operating_point(self):
        with pytest.warns(UserWarning) as record:
            detection_threshold_energy(0.9, 1e-6, 10.0, 1.0)
        message = str(record[0].message)
        assert 'optimistic by 3.49 dB' in message
        assert 'exact threshold 6.29 dB' in message

    @pytest.mark.parametrize('label, pd, pf, m', [
        ('guide DT_PASSIVE', 0.5, 1e-4, 500.0),
        ('guide DT_ACTIVE', 0.5, 1e-4, 50.0),
        ('test_detection_threshold_energy_formula', 0.9, 0.01, 100.0),
        ('test_five_db_per_decade', 0.5, 1e-4, 100.0),
        ('test_offset_from_urick', 0.5, 1e-4, 200.0),
        ('example_27', 0.5, 1e-4, 100.0),
    ])
    def test_no_documented_operating_point_warns(self, label, pd, pf, m):
        """A check that fires on the package's own worked examples trains
        users to ignore it. Every anchor the guide, the suite and the examples
        publish is inside the 1 dB promise, so every one must stay silent — and
        the assertion is on the measured error, not on the guard, so it also
        catches the envelope drifting off the promise."""
        error = (detection_threshold_energy(pd, pf, m, 1.0)
                 - self._exact_dt_db(pd, pf, m))
        assert abs(error) < 1.0, f'{label} is outside the promise: {error} dB'
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            detection_threshold_energy(pd, pf, m, 1.0)

    def test_the_fallback_bound_covers_an_unresolvable_operating_point(self):
        """The exact benchmark is allowed to fail; the check is not allowed to
        disappear when it does."""
        from uacpy.sonar.detection import _exact_detection_threshold_db
        # A sub-unity time-bandwidth product at a low Pf: the detection index
        # is a healthy 36, so this is not the degenerate pd == pf corner, but
        # the Gamma quantile ratio overflows and the exact value is not finite.
        pd, pf, m = 0.5, 1e-9, 1e-6
        assert sonar.detection_index(pd, pf) > 0
        assert not np.isfinite(_exact_detection_threshold_db(pd, pf, m))
        with pytest.warns(UserWarning, match='fitted fallback bound'):
            detection_threshold_energy(pd, pf, m, 1.0)

    def test_the_warning_does_not_change_the_returned_value(self):
        pd, pf, m = 0.9, 1e-6, 10.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got = detection_threshold_energy(pd, pf, m, 1.0)
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
            assert ts_cylinder(self.A, self.L, self.F,
                               angle_deg=ang) == pytest.approx(ref)

    def test_cylinder_main_lobe_is_the_documented_width(self):
        """The docstring promises a null-to-null width of ~lam/L radians, i.e.
        the first null where ``beta = pi`` (``sin th = lam/2L``)."""
        from uacpy.sonar.target_strength import ts_cylinder
        lam = self.C / self.F
        first_null = np.degrees(np.arcsin(lam / (2 * self.L)))
        # -309 dB at the analytic null (float64 sin() floor); -100 dB only
        # asserts that a null lands here, not how deep it is.
        assert ts_cylinder(self.A, self.L, self.F,
                           angle_deg=first_null) < -100.0
        assert 2 * first_null == pytest.approx(np.degrees(lam / self.L),
                                               rel=1e-3)

    def test_plate_at_normal_incidence_is_physical_optics(self):
        from uacpy.sonar.target_strength import ts_plate
        lam = self.C / self.F
        w, h = 4.0, 3.0
        assert ts_plate(w, h, self.F) == pytest.approx(20 * np.log10(w * h / lam))


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

    def test_the_far_field_is_silent_and_matches_the_closed_form(self):
        from uacpy.sonar.reverberation import boundary_reverberation
        r = np.array([500.0, 2000.0, 9000.0])
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got = boundary_reverberation(r, 200.0, -27.0, **self.KW)
        expected = (200.0 - 2 * 20 * np.log10(r) - 27.0
                    + 10 * np.log10(0.1 * r * (1500.0 * 0.1 / 2.0)))
        np.testing.assert_allclose(got, expected)
        assert rec == []

    def test_a_cell_longer_than_the_range_warns_with_the_error(self):
        from uacpy.sonar.reverberation import boundary_reverberation
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            boundary_reverberation(np.array([1.0, 10.0, 1000.0]), 200.0,
                                   -27.0, **self.KW)
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert 'c*tau/2 = 75 m' in msg and 'shortest 1 m' in msg

    def test_volume_reverberation_guards_the_same_domain(self):
        from uacpy.sonar.reverberation import volume_reverberation
        kw = dict(pulse_length_s=0.1, solid_angle_beamwidth_sr=0.01,
                  sound_speed=1500.0)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            volume_reverberation(np.array([1.0, 1000.0]), 200.0, -70.0, **kw)
        assert len(rec) == 1

    def test_a_zero_range_is_nan_without_a_numpy_warning(self):
        from uacpy.sonar.reverberation import boundary_reverberation
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
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
    * Etter Sect. 9.2 — "Chapman and Scott (1964) later validated these
      results over the frequency range 0.1 kHz to 6.4 kHz for grazing angle
      below 80 deg."

    The threshold follows JKPS, because that is the statement about the
    formula being RIGHT; Chapman & Scott's 80 deg is how far the data reach,
    which is a different claim and is only reported in the message.
    """

    def test_the_fitted_envelope_is_silent_and_matches_the_published_form(self):
        # Angles stay under the 40-50 deg JKPS says the formula performs well
        # within, and the wind under the 15 m/s the fit was measured over.
        from uacpy.sonar.scattering import chapman_harris_surface
        angles = np.array([10.0, 30.0, 45.0])
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got = chapman_harris_surface(angles, 10.0, 1000.0)
        assert rec == []
        # Independent transcription of the published form.
        beta = 158.0 * (10.0 * 1000.0 ** (1.0 / 3.0)) ** (-0.58)
        expected = (3.3 * beta * np.log10(angles / 30.0)
                    - 42.4 * np.log10(beta) + 2.6)
        np.testing.assert_allclose(got, expected)

    @pytest.mark.parametrize('freq', [100.0, 50000.0])
    def test_a_frequency_outside_the_fitted_band_warns(self, freq):
        from uacpy.sonar.scattering import chapman_harris_surface
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            chapman_harris_surface(10.0, 10.0, freq)
        hits = [w for w in rec if 'outside the 400-6400 Hz' in str(w.message)]
        assert len(hits) == 1

    @pytest.mark.parametrize('freq', [400.0, 6400.0])
    def test_the_band_edges_are_inclusive(self, freq):
        from uacpy.sonar.scattering import chapman_harris_surface
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            chapman_harris_surface(10.0, 10.0, freq)
        assert rec == []

    def test_a_grazing_angle_past_the_accuracy_limit_warns(self):
        from uacpy.sonar.scattering import chapman_harris_surface
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            chapman_harris_surface(np.array([10.0, 85.0]), 10.0, 1000.0)
        hits = [w for w in rec if 'grazing angle(s) exceed' in str(w.message)]
        assert len(hits) == 1
        msg = str(hits[0].message)
        assert 'steepest 85 deg' in msg
        # Both figures reach the caller, attributed to their own source.
        assert 'below 40-50 deg' in msg and '80 deg' in msg

    def test_the_threshold_is_the_jkps_accuracy_limit_not_the_data_range(self):
        # 60 deg is inside Chapman & Scott's 80 deg data range but past the
        # 40-50 deg JKPS says the formula performs well within.
        from uacpy.sonar.scattering import chapman_harris_surface
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            chapman_harris_surface(60.0, 10.0, 1000.0)
        assert [w for w in rec if 'grazing angle(s) exceed' in str(w.message)]

    def test_a_wind_speed_past_the_fitted_ceiling_warns(self):
        # JKPS Sect. 1.7.1 gives the fit a 15 m/s ceiling; nothing checked it.
        from uacpy.sonar.scattering import chapman_harris_surface
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            chapman_harris_surface(10.0, 40.0, 1000.0)      # 40 kn = 20.6 m/s
        hits = [w for w in rec if 'wind speed' in str(w.message)]
        assert len(hits) == 1 and '15 m/s' in str(hits[0].message)

    def test_a_wind_speed_inside_the_ceiling_is_silent(self):
        from uacpy.sonar.scattering import chapman_harris_surface
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            chapman_harris_surface(10.0, 25.0, 1000.0)      # 25 kn = 12.9 m/s
        assert [w for w in rec if 'wind speed' in str(w.message)] == []

    def test_the_two_envelope_checks_are_independent(self):
        from uacpy.sonar.scattering import chapman_harris_surface
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            chapman_harris_surface(85.0, 10.0, 50000.0)
        assert len(rec) == 2


class TestLambertBottomFlagsExtrapolation:
    """Every constant here is quoted from the corpus.

    * Etter Sect. 9.2, citing Urick (1983) Ch. 8: Lambert's law "appears to
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
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got = lambert_bottom(angles)
        assert rec == []
        expected = LAMBERT_MU_DB + 10.0 * np.log10(
            np.sin(np.deg2rad(angles)) ** 2)
        np.testing.assert_allclose(got, expected)

    def test_a_steeper_angle_warns_and_names_the_source(self):
        from uacpy.sonar.scattering import lambert_bottom
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
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
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
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
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
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
        from uacpy.acoustic_signal.arrays import sample_covariance
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
        from uacpy.acoustic_signal.arrays import sample_covariance
        d = self._snapshots()
        d[2, 3] = bad
        with pytest.raises(ConfigurationError, match='NaN or Inf'):
            sample_covariance(d)

    def test_the_non_finite_message_names_the_count_and_first_index(self):
        d = self._snapshots()
        d[2, 3] = NAN
        with pytest.raises(ConfigurationError) as exc:
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
        from uacpy.acoustic_signal.arrays import sample_covariance
        call = sonar.csdm if name == 'csdm' else sample_covariance
        with pytest.raises(ConfigurationError, match=cue) as exc:
            call(np.zeros(shape, dtype=complex))
        assert name in str(exc.value)

    def test_a_zero_sensor_covariance_never_reaches_a_surface(self):
        from uacpy.acoustic_signal.arrays import (bartlett_spectrum,
                                                  sample_covariance)
        with pytest.raises(ConfigurationError):
            bartlett_spectrum(sample_covariance(np.zeros((0, 5), complex)),
                              np.zeros((3, 0), complex))

    @pytest.mark.parametrize('name', ['csdm', 'sample_covariance'])
    def test_a_field_is_named_with_the_bridge_rather_than_a_cast_error(
            self, name):
        """``np.asarray(Field, dtype=complex)`` raises ``must be real number,
        not Field``, which names nothing the caller passed. Both covariance
        names give the typed message the rest of the package gives."""
        from uacpy.acoustic_signal.arrays import sample_covariance
        from uacpy.core.results import Field
        call = sonar.csdm if name == 'csdm' else sample_covariance
        t = np.arange(64) / 64.0
        field = Field(data=np.sin(2 * np.pi * 4 * t), coords={'time': t})
        with pytest.raises(ConfigurationError) as exc:
            call(field)
        message = str(exc.value)
        assert name in message
        assert 'Field.data' in message

    def test_a_one_sensor_one_snapshot_matrix_is_the_admissible_corner(self):
        """Both sides of both new boundaries: ``(1, 1)`` is the smallest
        matrix that still defines a covariance, and it is accepted."""
        from uacpy.acoustic_signal.arrays import sample_covariance
        d = np.array([[2.0 + 1.0j]])
        np.testing.assert_allclose(sonar.csdm(d), d @ d.conj().T)
        np.testing.assert_allclose(sample_covariance(d), d @ d.conj().T)

    def test_an_all_finite_snapshot_set_beamforms(self):
        # The negative control for the guard: the ordinary path is untouched,
        # and the surface it produces carries no NaN.
        rng = np.random.default_rng(11)
        d = self._snapshots()
        replicas = (rng.standard_normal((6, 5))
                    + 1j * rng.standard_normal((6, 5)))
        surface = sonar.bartlett(sonar.csdm(d), replicas)
        assert np.all(np.isfinite(surface))

    def test_a_nan_snapshot_never_reaches_the_ambiguity_surface(self):
        rng = np.random.default_rng(11)
        d = self._snapshots()
        d[0, 0] = NAN
        replicas = (rng.standard_normal((6, 5))
                    + 1j * rng.standard_normal((6, 5)))
        for processor in (sonar.bartlett, sonar.mvdr):
            with pytest.raises(ConfigurationError):
                processor(sonar.csdm(d), replicas)


class TestSonarPositivityGuardsRefuseNaN:
    @pytest.mark.parametrize("kwargs", [
        {'wind_speed_kn': NAN, 'frequency': 1000.0},
        {'wind_speed_kn': 10.0, 'frequency': NAN},
    ])
    def test_chapman_harris_nan_wind_or_frequency_raises(self, kwargs):
        with pytest.raises(ConfigurationError, match="must be > 0 and finite"):
            chapman_harris_surface(10.0, **kwargs)

    def test_chapman_harris_nan_grazing_angle_warns(self):
        # The grazing-angle guard warns rather than raising (a 0 deg angle is a
        # legitimate -inf), so NaN must reach the same warning a negative does.
        with pytest.warns(UserWarning, match="negative or non-finite"):
            out = chapman_harris_surface(NAN, 10.0, 1000.0)
        assert np.isnan(out)

    def test_chapman_harris_zero_grazing_angle_stays_minus_inf_unwarned(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.isneginf(chapman_harris_surface(0.0, 10.0, 1000.0))

    def test_column_scattering_nan_thickness_raises(self):
        with pytest.raises(ConfigurationError, match="must be > 0 and finite"):
            column_scattering_strength(-30.0, NAN)

    @pytest.mark.parametrize("call", [
        lambda: ts.ts_sphere(NAN),
        lambda: ts.ts_cylinder(NAN, 2.0, 1000.0),
        lambda: ts.ts_plate(1.0, 1.0, NAN),
        lambda: ts.ts_ellipsoid(2.0, 1.0, NAN),
    ])
    def test_target_strength_nan_dimension_or_frequency_raises(self, call):
        with pytest.raises(ConfigurationError, match="must be > 0 and finite"):
            call()

    @pytest.mark.parametrize("call", [
        lambda: ts.ts_sphere(INF),
        lambda: ts.ts_cylinder(INF, 2.0, 1000.0),
        lambda: ts.ts_plate(1.0, 1.0, INF),
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
        with pytest.raises(ConfigurationError, match=r"pf must be in \(0, 1\)"):
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
    ``_carrier_validate._require_positive``'s docstring; what lives here is the
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
        from uacpy.acoustic_signal._signal_validate import (
            require_positive_finite_scalar)
        from uacpy.core._carrier_validate import _require_positive as carrier
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
            with pytest.raises(ConfigurationError):
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
        from uacpy.core._carrier_validate import _require_positive as carrier
        for name in ('acoustic_signal', 'sonar'):
            with pytest.raises(ConfigurationError):
                self._guards()[name](value)
        with pytest.raises((TypeError, ValueError)):
            carrier(value, 'x')

    def test_the_scalar_guards_return_the_value_as_a_float(self):
        """The two scalar guards are used as ``x = guard(x)``; the carrier one
        validates in place and returns ``None``. Pinned so a consolidation
        cannot swap one contract for the other silently."""
        from uacpy.acoustic_signal._signal_validate import (
            require_positive_finite_scalar)
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
     lambda: chapman_harris_surface(10.0, INF, 1000.0)),
    ('chapman_harris_surface/frequency',
     lambda: chapman_harris_surface(10.0, 10.0, INF)),
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
            out = chapman_harris_surface(INF, 10.0, 1000.0)
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
    def test_a_db_level_keeps_its_non_finite_meaning(self, level):
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
        for db_arg in (lambda v: sonar.boundary_reverberation(
                           [100.0, 200.0], v, -30.0, pulse_length_s=0.01,
                           horizontal_beamwidth_rad=0.1),
                       lambda v: sonar.column_scattering_strength(v, 10.0)):
            assert not np.all(np.isfinite(np.asarray(db_arg(level), float)))

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
            assert np.isneginf(chapman_harris_surface(0.0, 10.0, 1000.0))
            assert np.isfinite(chapman_harris_surface(10.0, 10.0, 1000.0))
        assert np.all(np.isfinite(roc_curve(0.0)[1]))
        assert np.isfinite(detection_threshold_energy(0.9, 1e-4, 100.0, 1.0))
        assert np.isfinite(column_scattering_strength(-30.0, 10.0))


# ── detection_range far-edge recovery ───────────────────────────────────────
#
# `detection_range` returns the last sampled range when the signal excess dips
# negative inside the grid and recovers at the far edge. The value is finite,
# so the `np.isfinite` test the sonar guide recommends accepts it as a modelled
# crossing. Measured on a Kraken shelf budget: 20000.0 m on a 20 km grid
# against 45947 m on a 120 km grid, a factor 2.30. The return value is pinned
# elsewhere in this file; what is added here is the warning beside it.
#
# Every threshold below is pinned on BOTH sides: a mutation campaign found that
# a guard test using values far from the boundary pins that the guard fires,
# never where.

def _messages(fn, needle):
    """Run ``fn`` and return the warning messages containing ``needle``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        fn()
    return [str(w.message) for w in rec if needle in str(w.message)]


_LOWER_BOUND = 'LOWER BOUND'


class TestDetectionRangeFarEdgeRecovery:
    def test_far_edge_recovery_warns_the_value_is_a_lower_bound(self):
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        se = np.array([5.0, -1.0, 2.0, 3.0, 4.0])
        msgs = _messages(lambda: sonar.detection_range(r, se), _LOWER_BOUND)
        assert len(msgs) == 1
        assert 'widen receiver.ranges' in msgs[0]

    def test_far_edge_recovery_returns_the_last_sampled_range(self):
        """Round 19 pinned the return value; CS1 adds a warning beside it,
        not a different answer."""
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        se = np.array([5.0, -1.0, 2.0, 3.0, 4.0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert sonar.detection_range(r, se) == pytest.approx(4.0)

    def test_negative_at_the_last_sample_is_silent(self):
        """One sample either side of the branch: SE at the outermost range is
        -1 dB here and +1 dB in the sibling test, and only the second is a
        bound rather than a crossing."""
        r = np.array([0.0, 1.0, 2.0, 3.0])
        se = np.array([5.0, -1.0, 1.0, -1.0])
        msgs = _messages(lambda: sonar.detection_range(r, se), _LOWER_BOUND)
        assert msgs == []

    def test_positive_at_the_last_sample_warns(self):
        r = np.array([0.0, 1.0, 2.0, 3.0])
        se = np.array([5.0, -1.0, 1.0, 1.0])
        msgs = _messages(lambda: sonar.detection_range(r, se), _LOWER_BOUND)
        assert len(msgs) == 1

    def test_zero_at_the_last_sample_warns(self):
        """``positive`` is ``se >= 0``, so SE == 0 at the far edge is the
        warning side of that comparison, not the silent one."""
        r = np.array([0.0, 1.0, 2.0, 3.0])
        se = np.array([5.0, -1.0, 1.0, 0.0])
        msgs = _messages(lambda: sonar.detection_range(r, se), _LOWER_BOUND)
        assert len(msgs) == 1

    def test_positive_everywhere_is_silent(self):
        """``inf`` already says "beyond the grid"; the adjacent case is the
        same vector with one sample pushed negative, which does warn."""
        r = np.linspace(1.0, 10.0, 10)
        always = np.ones(10)
        assert _messages(lambda: sonar.detection_range(r, always),
                         _LOWER_BOUND) == []
        dipped = always.copy()
        dipped[4] = -1.0
        assert len(_messages(lambda: sonar.detection_range(r, dipped),
                             _LOWER_BOUND)) == 1

    def test_negative_everywhere_is_silent(self):
        r = np.linspace(1.0, 10.0, 10)
        assert _messages(lambda: sonar.detection_range(r, -np.ones(10)),
                         _LOWER_BOUND) == []

    def test_no_data_hole_before_a_negative_far_edge_is_silent(self):
        r = np.array([0.0, 1000.0, 2000.0])
        se = np.array([5.0, np.nan, -5.0])
        assert _messages(lambda: sonar.detection_range(r, se),
                         _LOWER_BOUND) == []

    def test_trailing_no_data_after_a_recovery_warns_about_the_last_filled_cell(self):
        """The masked array's last entry is the outermost cell the model
        FILLED, so the bound is that range, not the grid's edge."""
        r = np.array([0.0, 1.0, 2.0, 3.0])
        se = np.array([5.0, -1.0, 2.0, np.nan])
        msgs = _messages(lambda: sonar.detection_range(r, se), _LOWER_BOUND)
        assert len(msgs) == 1
        assert '(2 m)' in msgs[0]


class TestSignalExcessFieldRejectsANonScalarNoiseLevel:
    """``reverberation_level`` is the one per-range term of this budget; the
    rest are documented scalars. An array ``noise_level`` computed the whole
    signal-excess field and *then* raised ``TypeError: only 0-dimensional
    arrays can be converted to Python scalars`` from the budget dict, naming
    neither the function nor the argument."""

    @staticmethod
    def _tl_field():
        from uacpy.core.results import Field
        return Field(data=np.full((5, 7), 60.0),
                     coords={'depth': np.linspace(0.0, 100.0, 5),
                             'range': np.linspace(100.0, 5000.0, 7)})

    @pytest.mark.parametrize('bad', ['per_range', 'two_d'])
    @pytest.mark.parametrize('mode', ['passive', 'active'])
    def test_an_array_noise_level_is_named_up_front(self, mode, bad):
        field = self._tl_field()
        nl = (np.linspace(55.0, 65.0, 7) if bad == 'per_range'
              else np.zeros((5, 7)))
        kwargs = ({'source_level': 200.0} if mode == 'passive'
                  else {'source_level': 200.0, 'target_strength': -20.0})
        call = getattr(sonar, f'{mode}_signal_excess_field')
        with pytest.raises(ConfigurationError) as exc:
            call(field, noise_level=nl, **kwargs)
        message = str(exc.value)
        assert f'{mode}_signal_excess_field' in message
        assert 'noise_level' in message
        assert 'reverberation_level' in message

    def test_a_scalar_noise_level_is_accepted(self):
        se = sonar.passive_signal_excess_field(
            self._tl_field(), source_level=200.0, noise_level=60.0)
        assert se.data.shape == (5, 7)

    def test_the_per_range_term_the_message_points_at_accepts_an_array(
            self):
        se = sonar.active_signal_excess_field(
            self._tl_field(), source_level=200.0, target_strength=-20.0,
            noise_level=60.0,
            reverberation_level=np.linspace(55.0, 65.0, 7))
        assert se.data.shape == (5, 7)


class TestScalarSonarEquationNamesItsFieldTwin:
    """Handing a ``Field`` to the array-taking sonar-equation functions
    reached ``float()`` and raised ``TypeError: float() argument must be a
    string or a real number, not 'Field'`` — while the working call is one
    suffix away and the message never said so."""

    @staticmethod
    def _tl_field():
        from uacpy.core.results import Field
        return Field(data=np.full((5, 7), 60.0),
                     coords={'depth': np.linspace(0.0, 100.0, 5),
                             'range': np.linspace(100.0, 5000.0, 7)})

    def test_passive_signal_excess_names_passive_signal_excess_field(self):
        with pytest.raises(ConfigurationError) as exc:
            sonar.passive_signal_excess(180.0, self._tl_field(), 60.0)
        message = str(exc.value)
        assert 'passive_signal_excess_field' in message

    def test_active_signal_excess_names_active_signal_excess_field(self):
        with pytest.raises(ConfigurationError) as exc:
            sonar.active_signal_excess(180.0, self._tl_field(), -20.0,
                                       noise_level=60.0)
        assert 'active_signal_excess_field' in str(exc.value)

    def test_the_twin_the_message_names_accepts_the_same_field(self):
        se = sonar.passive_signal_excess_field(
            self._tl_field(), source_level=180.0, noise_level=60.0)
        assert se.data.shape == (5, 7)

    def test_a_plain_array_reaches_the_scalar_function(self):
        out = sonar.passive_signal_excess(180.0, np.full((5, 7), 60.0), 60.0)
        assert np.asarray(out).shape == (5, 7)
