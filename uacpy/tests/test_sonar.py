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

    def test_di_array_with_zero_plus_ag_still_raises(self):
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
        # Cylinder/plate geometric formulas hold for ka > 1 (Urick's
        # ka >> 1; Abraham's reference cylinder runs at ka ~ 4).
        # a = 0.05 m at 1 kHz -> ka = 2*pi*1000/1500*0.05 = 0.209 < 1: warn.
        with pytest.warns(UserWarning, match="geometric"):
            sonar.ts_cylinder(0.05, 5.0, 1000.0)
        # Plate checks the smaller dimension: min(0.1, 0.2) -> ka = 0.419.
        with pytest.warns(UserWarning, match="geometric"):
            sonar.ts_plate(0.1, 0.2, 1000.0)
        # ka > 1 stays silent: a = 0.5 m -> ka = 2.09; 1 m plate -> ka = 4.19.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            sonar.ts_cylinder(0.5, 5.0, 1000.0)
            sonar.ts_plate(1.0, 1.0, 1000.0)

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
