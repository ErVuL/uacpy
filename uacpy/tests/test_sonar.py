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
        # Independent hand value (TL=60, cell=7500 m², 10log10(7500)=38.751):
        # 220 - 120 - 40 + 38.751 = 98.751 dB — anchors the formula, not just
        # re-derives it.
        assert rl[0] == pytest.approx(98.751, abs=0.01)

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
        # Independent hand value: 10log10(1e8 + 2.5119e7) = 80.973 dB.
        assert tot[0] == pytest.approx(80.973, abs=0.01)

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

    def test_albersheim_reasonable(self):
        # Single-pulse Pd=0.5, Pf=1e-4 -> ~9.4 dB.
        assert sonar.albersheim_snr(0.5, 1e-4) == pytest.approx(9.4, abs=0.3)

    def test_albersheim_integration_lowers_snr(self):
        assert sonar.albersheim_snr(0.9, 1e-6, 10) < sonar.albersheim_snr(0.9, 1e-6, 1)

    def test_detection_threshold_energy_formula(self):
        dt = sonar.detection_threshold_energy(0.9, 0.01, 100.0, 1.0)
        d = sonar.detection_index(0.9, 0.01)
        # DT = 5*log10(d / (w*t)) (Urick energy detector); anchor to the value.
        assert dt == pytest.approx(5 * np.log10(d / (100.0 * 1.0)))

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
        assert ts_null < -60.0  # deep pattern null
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
