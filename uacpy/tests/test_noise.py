"""
Smoke tests for the simplified uacpy.noise module.

Covers ``compute_windnoise`` and the ``WenzNoise`` class
(wind / shipping / rain / thermal / turbulence).
"""

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError
from uacpy.noise import compute_windnoise, WenzNoise


@pytest.fixture
def freqs():
    # Log-spaced 1 Hz → 100 kHz so every Wenz band has plenty of points.
    return np.logspace(0.0, 5.0, 200)


# ─── compute_windnoise ───────────────────────────────────────────────────────


def test_compute_windnoise_zero_wind(freqs):
    """``u == 0`` silences the surface-noise source: spectral level is
    ``-inf`` dB at every frequency so the incoherent dB sum with the
    other Wenz components drops wind cleanly."""
    NL = compute_windnoise(freqs, u=0)
    assert NL.shape == freqs.shape
    assert np.all(np.isneginf(NL))


def test_compute_windnoise_scalar_frequency():
    """``compute_windnoise(scalar_f, u)`` returns a 1-element 1-D array."""
    NL = compute_windnoise(100.0, u=10, water_depth='deep')
    assert NL.shape == (1,)
    assert np.isfinite(NL[0])


def test_compute_windnoise_negative_wind_raises():
    """Negative wind speed raises :class:`ConfigurationError`."""
    with pytest.raises(ConfigurationError, match="non-negative"):
        compute_windnoise(np.array([100.0]), u=-5, water_depth='deep')


def test_compute_windnoise_increases_with_wind(freqs):
    low = compute_windnoise(freqs, u=5,  water_depth='deep')
    high = compute_windnoise(freqs, u=25, water_depth='deep')
    assert np.all(np.isfinite(low))
    assert np.all(np.isfinite(high))
    band = (freqs >= 100) & (freqs <= 1500)
    assert np.mean(high[band]) > np.mean(low[band])


def test_compute_windnoise_shallow_louder_than_deep(freqs):
    deep = compute_windnoise(freqs, u=10, water_depth='deep')
    shallow = compute_windnoise(freqs, u=10, water_depth='shallow')
    band = (freqs >= 100) & (freqs <= 1500)
    assert np.mean(shallow[band]) > np.mean(deep[band])


def test_compute_windnoise_band_integrate(freqs):
    """``band_integrate=True`` returns the band SPL: the spectral level plus
    ``10·log10(Δf)``, where the band edges sit at the midpoints between
    consecutive frequencies and the two end bands span only the half-spacing
    to their single neighbour."""
    pointwise = compute_windnoise(freqs, u=10, water_depth='deep')
    integrated = compute_windnoise(freqs, u=10, water_depth='deep', band_integrate=True)
    assert pointwise.shape == freqs.shape
    assert integrated.shape == freqs.shape
    assert np.all(np.isfinite(integrated))
    mids = (freqs[1:] + freqs[:-1]) / 2
    edges = np.concatenate(([freqs[0]], mids, [freqs[-1]]))
    df = edges[1:] - edges[:-1]
    np.testing.assert_allclose(integrated, pointwise + 10 * np.log10(df),
                               rtol=0, atol=1e-9)
    # Octave grid anchor: the 400 Hz band spans 300-600 Hz, so its band SPL
    # is the spectral level plus 10·log10(300) = +24.771 dB.
    f5 = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
    spec = compute_windnoise(f5, u=10, water_depth='deep')
    band = compute_windnoise(f5, u=10, water_depth='deep', band_integrate=True)
    assert band[2] == pytest.approx(spec[2] + 10 * np.log10(300.0), abs=1e-9)


def test_compute_windnoise_pinned_anchors():
    """Pin the wind-noise spectral level at four (wind, frequency) anchors.

    The values are DRDC-RDDC-2022-D051 eqs. (8)-(16) evaluated by hand for
    deep water (c0 = 42): they agree with ``compute_windnoise`` to ~1e-11 dB,
    so ``abs=1e-6`` is a float-noise tolerance, not a fitted margin. The
    equation-level check lives in
    ``test_noise_submodels.py::test_wind_follows_drdc_annex_a_below_the_cutoff``;
    these anchors additionally freeze the numbers, so a coefficient edit
    cannot pass by changing test and code together.
    """
    expected = {
        (10.0, 100.0): 58.9076655148,
        (10.0, 1000.0): 59.6619005873,
        (20.0, 100.0): 65.3631715344,
        (20.0, 1000.0): 65.6516662333,
    }
    for (u, fhz), level in expected.items():
        got = compute_windnoise(np.array([fhz]), u=u, water_depth='deep')
        assert got[0] == pytest.approx(level, abs=1e-6)


def test_wenznoise_high_freq_only_no_rain_meld_crash():
    """Rain melding only fires when both <7kHz and >7kHz frequencies exist."""
    f_high = np.logspace(4.0, 5.0, 30)  # all > 7 kHz
    wenz = WenzNoise(f_high, wind_speed=10, rain_rate='moderate')
    assert np.all(np.isfinite(wenz.rain))


# ─── WenzNoise — invariants ─────────────────────────────────────────────────


def test_wenznoise_default_attributes(freqs):
    """Inactive bands are carried as ``-inf`` in each component array;
    ``.total`` stays finite as long as at least one source is active
    at every frequency."""
    wenz = WenzNoise(freqs, wind_speed=10)
    for attr in ('total', 'shipping', 'wind', 'rain', 'thermal', 'turbulence'):
        v = getattr(wenz, attr)
        assert v.shape == freqs.shape
    # The total is the incoherent sum — must be finite when any of the
    # five components is active at each frequency (it is, for these inputs).
    assert np.all(np.isfinite(wenz.total))
    # Components are allowed to be -inf when inactive but never NaN.
    for attr in ('shipping', 'wind', 'rain', 'thermal', 'turbulence'):
        v = getattr(wenz, attr)
        assert not np.any(np.isnan(v)), f"{attr} contains NaN"


def test_wenznoise_constructor_defaults():
    """Constructor defaults are ``shipping_level='medium'``, ``rain_rate='no'``,
    ``water_depth='deep'``; a default-constructed spectrum equals the fully
    explicit one and carries rain switched off (``-inf``)."""
    import inspect
    sig = inspect.signature(WenzNoise.__init__)
    assert sig.parameters['shipping_level'].default == 'medium'
    assert sig.parameters['rain_rate'].default == 'no'
    assert sig.parameters['water_depth'].default == 'deep'
    f = np.logspace(1.0, 4.0, 50)
    default = WenzNoise(f, wind_speed=10)
    explicit = WenzNoise(f, wind_speed=10, shipping_level='medium',
                         rain_rate='no', water_depth='deep')
    np.testing.assert_array_equal(default.total, explicit.total)
    assert np.all(np.isneginf(default.rain))


def test_wenznoise_components_named(freqs):
    wenz = WenzNoise(freqs, wind_speed=10, shipping_level='medium',
                     rain_rate='moderate')
    c = wenz.components
    assert c._fields == ('total', 'wind', 'shipping', 'rain',
                         'thermal', 'turbulence')
    assert len(c) == 6
    np.testing.assert_array_equal(c.total, wenz.total)
    np.testing.assert_array_equal(c.shipping, wenz.shipping)
    np.testing.assert_array_equal(c.wind, wenz.wind)
    np.testing.assert_array_equal(c.rain, wenz.rain)
    np.testing.assert_array_equal(c.thermal, wenz.thermal)
    np.testing.assert_array_equal(c.turbulence, wenz.turbulence)


def test_wenznoise_total_geq_components(freqs):
    wenz = WenzNoise(freqs, wind_speed=10, shipping_level='medium',
                     rain_rate='moderate')
    components = np.column_stack([wenz.shipping, wenz.wind, wenz.rain,
                                  wenz.thermal, wenz.turbulence])
    # Total in dB must be ≥ each individual component everywhere.
    assert np.all(wenz.total + 1e-6 >= components.max(axis=1))


def test_wenznoise_shipping_levels_ordered(freqs):
    band = (freqs >= 30) & (freqs <= 200)
    low = WenzNoise(freqs, wind_speed=5, shipping_level='low').total[band]
    med = WenzNoise(freqs, wind_speed=5, shipping_level='medium').total[band]
    high = WenzNoise(freqs, wind_speed=5, shipping_level='high').total[band]
    assert np.mean(low) < np.mean(med) < np.mean(high)


def test_wenznoise_as_psd_round_trip(freqs):
    wenz = WenzNoise(freqs, wind_speed=10)
    # Default returns µPa²/Hz — the same 1 µPa reference as .total
    # (dB re 1 µPa²/Hz), so a plain 10·log10 reproduces .total exactly.
    upa2 = wenz.as_psd()
    np.testing.assert_allclose(10 * np.log10(upa2), wenz.total, rtol=0, atol=1e-9)
    # An explicit SI request (ref = 1 µPa in Pa) returns Pa²/Hz.
    pa2 = wenz.as_psd(ref=1e-6)
    np.testing.assert_allclose(10 * np.log10(pa2 / 1e-12), wenz.total, rtol=0, atol=1e-9)


def test_wenznoise_repr_contains_params(freqs):
    wenz = WenzNoise(freqs, wind_speed=15, water_depth='shallow',
                     shipping_level='high', rain_rate='heavy')
    s = repr(wenz)
    assert 'wind=15' in s
    assert "'shallow'" in s
    assert "'high'" in s
    assert "'heavy'" in s


def test_wenznoise_rejects_invalid_kwargs(freqs):
    with pytest.raises(ConfigurationError, match='water_depth'):
        WenzNoise(freqs, wind_speed=10, water_depth='abyssal')
    with pytest.raises(ConfigurationError, match='shipping_level'):
        WenzNoise(freqs, wind_speed=10, shipping_level='extreme')
    with pytest.raises(ConfigurationError, match='rain_rate'):
        WenzNoise(freqs, wind_speed=10, rain_rate='monsoon')


def test_wenznoise_rejects_dc_and_negative_frequencies():
    # The empirical fits are all log10(f); a DC bin (common from a raw rfft
    # grid) would otherwise produce log10(0) = -inf/NaN, not a clear error.
    with pytest.raises(ConfigurationError, match='> 0 Hz'):
        WenzNoise(np.array([0.0, 10.0, 100.0]), wind_speed=10)
    with pytest.raises(ConfigurationError, match='> 0 Hz'):
        WenzNoise(np.array([-5.0, 10.0]), wind_speed=10)


def test_wenznoise_plot_returns_fig_ax(freqs):
    wenz = WenzNoise(freqs, wind_speed=15)
    from uacpy.visualization import plot_wenz
    fig, ax = plot_wenz(wenz)
    assert fig is not None and ax is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_wenznoise_plot_total_only(freqs):
    wenz = WenzNoise(freqs, wind_speed=15)
    from uacpy.visualization import plot_wenz
    fig, ax = plot_wenz(wenz, show_components=False)
    assert fig is not None and ax is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_wenznoise_total_matches_linear_sum_of_components(freqs):
    """``total`` equals the incoherent linear sum of the five
    components within numerical precision; ``-inf`` components
    contribute zero linear power."""
    wenz = WenzNoise(freqs, wind_speed=10, shipping_level='medium',
                     rain_rate='moderate', water_depth='deep')
    # Treat -inf rigorously: 10**(-inf/10) = 0.
    lin_sum = (
        10.0 ** (wenz.shipping / 10.0)
        + 10.0 ** (wenz.wind / 10.0)
        + 10.0 ** (wenz.rain / 10.0)
        + 10.0 ** (wenz.thermal / 10.0)
        + 10.0 ** (wenz.turbulence / 10.0)
    )
    expected = 10.0 * np.log10(lin_sum)
    np.testing.assert_allclose(wenz.total, expected, rtol=1e-10, atol=1e-10)


def test_wenznoise_zero_wind_drops_wind_from_total(freqs):
    """``wind_speed=0`` makes the wind component ``-inf`` so it drops out
    of ``.total``; the total equals the sum of the four remaining
    components."""
    wenz = WenzNoise(freqs, wind_speed=0, shipping_level='medium',
                     rain_rate='moderate', water_depth='deep')
    assert np.all(np.isneginf(wenz.wind))
    lin_no_wind = (
        10.0 ** (wenz.shipping / 10.0)
        + 10.0 ** (wenz.rain / 10.0)
        + 10.0 ** (wenz.thermal / 10.0)
        + 10.0 ** (wenz.turbulence / 10.0)
    )
    expected = 10.0 * np.log10(lin_no_wind)
    np.testing.assert_allclose(wenz.total, expected, rtol=1e-10, atol=1e-10)


def test_thermal_overtakes_wind_near_110_khz_at_10_knots():
    """The Mellen thermal floor (``-75 + 20·log10 f``, rising 20 dB/decade)
    crosses above the 10 kn wind curve (falling 16.6 dB/decade above 2 kHz) at
    112.6 kHz — the two closed forms intersect at
    ``log10 f = (Lw,2000 + 16.6096·log10 2000 + 75) / 36.6096``. In a 1 kn
    calm the crossover drops to 32.0 kHz."""
    f = np.logspace(5.0, 5.1, 201)                # 100.0 - 125.9 kHz
    wenz = WenzNoise(f, wind_speed=10.0, shipping_level='no')
    d = wenz.thermal - wenz.wind
    crossing = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
    assert crossing.size == 1
    assert 105e3 < f[crossing[0]] < 120e3
    f_calm = np.logspace(4.45, 4.6, 201)          # 28.2 - 39.8 kHz
    calm = WenzNoise(f_calm, wind_speed=1.0, shipping_level='no')
    d_calm = calm.thermal - calm.wind
    crossing_calm = np.flatnonzero((d_calm[:-1] < 0) & (d_calm[1:] >= 0))
    assert crossing_calm.size == 1
    assert 30e3 < f_calm[crossing_calm[0]] < 34e3


class TestShipRadiatedNoise:
    """ISO 17208 radiated noise level + monopole source level."""

    def test_rnl_spherical_spreading(self):
        from uacpy.noise import radiated_noise_level
        # RNL = received SPL + 20 log10(r)
        assert radiated_noise_level(120.0, 100.0) == pytest.approx(120.0 + 40.0)
        assert radiated_noise_level(120.0, 1.0) == pytest.approx(120.0)

    def test_nominal_source_depth(self):
        from uacpy.noise import nominal_source_depth
        assert nominal_source_depth(10.0) == pytest.approx(7.0)   # 0.7 * draught

    def test_lloyd_mirror_high_frequency_asymptote(self):
        from uacpy.noise import lloyd_mirror_correction
        # high f -> incoherent source+image -> -10 log10(2) = -3.01 dB
        assert lloyd_mirror_correction(1e5, 7.0) == pytest.approx(-3.0103, abs=1e-3)

    def test_lloyd_mirror_low_frequency_positive(self):
        from uacpy.noise import lloyd_mirror_correction
        # surface dipole suppresses low-f radiation -> MSL >> RNL -> large +ve
        assert lloyd_mirror_correction(10.0, 7.0) > 5.0

    def test_monopole_source_level_is_rnl_plus_correction(self):
        from uacpy.noise import (monopole_source_level, radiated_noise_level,
                                 lloyd_mirror_correction)
        rnl = radiated_noise_level(120.0, 150.0)
        ds, f = 7.0, 125.0
        assert monopole_source_level(rnl, f, ds) == pytest.approx(
            rnl + lloyd_mirror_correction(f, ds))

    def test_rnl_uncertainty_db_pinned(self):
        """ISO 17208-2 §5 combined RNL measurement uncertainty per band group:
        5 dB (10-100 Hz), 3 dB (125 Hz-16 kHz), 4 dB (>20 kHz)."""
        from uacpy.noise import RNL_UNCERTAINTY_DB
        assert RNL_UNCERTAINTY_DB == {"low": 5.0, "mid": 3.0, "high": 4.0}

    def test_lloyd_mirror_low_frequency_anchors(self):
        """ISO 17208-2 Formula 3 at 10 Hz, c = 1500 m/s: ΔL = +12.595 dB for
        the 5.6 m nominal source depth of an 8 m draught (0.7·8), +18.615 dB
        at 2.8 m and +7.146 dB at 10.5 m."""
        from uacpy.noise import lloyd_mirror_correction, nominal_source_depth
        d_s = nominal_source_depth(8.0)
        assert d_s == pytest.approx(5.6)
        assert lloyd_mirror_correction(10.0, d_s) == pytest.approx(12.5954, abs=1e-3)
        assert lloyd_mirror_correction(10.0, 2.8) == pytest.approx(18.6151, abs=1e-3)
        assert lloyd_mirror_correction(10.0, 10.5) == pytest.approx(7.1457, abs=1e-3)

    def test_lloyd_mirror_dip(self):
        """ΔL dips to −4.0708 dB at kd = 2.835 before settling on the
        −3.01 dB asymptote; at d = 5.6 m that kd falls at
        2.835·1500/(2π·5.6) = 120.9 Hz. The dip value depends on kd alone, so
        it is identical at every depth."""
        from uacpy.noise import lloyd_mirror_correction
        import numpy as np
        f = np.linspace(60.0, 400.0, 1701)
        v = lloyd_mirror_correction(f, 5.6)
        i = int(np.argmin(v))
        assert v[i] == pytest.approx(-4.0708, abs=1e-3)
        assert f[i] == pytest.approx(120.9, abs=1.0)
        v_deep = lloyd_mirror_correction(np.linspace(30.0, 120.0, 1001), 10.5)
        assert v_deep.min() == pytest.approx(-4.0708, abs=1e-3)


class TestMarineMammalWeighting:
    """Southall et al. 2019 auditory weighting functions (Table 5)."""

    ALL_GROUPS = ("LF", "HF", "VHF", "SI", "PCW", "OCW", "PCA", "OCA")

    def test_weighting_params_cover_all_eight_groups(self):
        """Southall Table 5 defines exactly eight hearing groups — the six
        in-water ones plus the in-air PCA and OCA — each carrying the five
        curve parameters and the published exposure constant K."""
        from uacpy.noise import WEIGHTING_PARAMS, HEARING_GROUPS
        assert set(WEIGHTING_PARAMS) == set(self.ALL_GROUPS)
        assert set(HEARING_GROUPS) == set(self.ALL_GROUPS)
        for params in WEIGHTING_PARAMS.values():
            assert set(params) == {"a", "b", "f1", "f2", "C", "K"}

    def test_peak_is_zero_db(self):
        from uacpy.noise import auditory_weighting
        import numpy as np
        f = np.logspace(1, 5.5, 5000)
        for g in self.ALL_GROUPS:
            assert auditory_weighting(f, g).max() == pytest.approx(0.0, abs=0.02)

    def test_low_frequency_slope_is_20a(self):
        from uacpy.noise import auditory_weighting, WEIGHTING_PARAMS
        for g in self.ALL_GROUPS:
            a = WEIGHTING_PARAMS[g]["a"]
            f1 = WEIGHTING_PARAMS[g]["f1"] * 1000.0
            slope = auditory_weighting(0.1 * f1, g) - auditory_weighting(0.01 * f1, g)
            assert slope == pytest.approx(20.0 * a, abs=0.3)   # +20a dB/decade

    def test_weighting_anchors_at_30_hz(self):
        """Southall Table 5 closed form at the 30 Hz shipping peak:
        LF −16.445 dB, VHF −92.314 dB — a 75.87 dB spread between the two
        groups' reception of the same band."""
        from uacpy.noise import auditory_weighting
        lf = auditory_weighting(30.0, "LF")
        vhf = auditory_weighting(30.0, "VHF")
        assert lf == pytest.approx(-16.4448, abs=1e-3)
        assert vhf == pytest.approx(-92.3142, abs=1e-3)
        assert lf - vhf == pytest.approx(75.8694, abs=1e-2)

    def test_weighted_broadband_levels_of_the_guide_spectrum(self):
        """The guide §8 spectrum (10 kn, medium shipping, 10 Hz-100 kHz on a
        1200-point log grid) integrates to 97.577 dB re 1 µPa² unweighted,
        94.029 dB LF-weighted and 83.510 dB VHF-weighted. The values are the
        independent DRDC + Southall closed forms; the guide text prints
        97.7 / 94.0 / 83.5, and its unweighted 97.7 is 0.12 dB above what the
        composite produces on its own stated grid (94.0 and 83.5 agree)."""
        from uacpy.noise import weighted_level
        import numpy as np
        f = np.logspace(1.0, 5.0, 1200)
        wenz = WenzNoise(f, wind_speed=10.0, shipping_level='medium')
        unweighted = 10.0 * np.log10(np.trapezoid(10.0 ** (wenz.total / 10.0), f))
        assert unweighted == pytest.approx(97.5772, abs=5e-3)
        assert weighted_level(wenz.total, f, "LF") == pytest.approx(94.0292, abs=5e-3)
        assert weighted_level(wenz.total, f, "VHF") == pytest.approx(83.5095, abs=5e-3)

    def test_unknown_group_raises(self):
        from uacpy.noise import auditory_weighting
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            auditory_weighting(1000.0, "MF")

    def test_apply_and_weighted_level(self):
        from uacpy.noise import apply_weighting, weighted_level, auditory_weighting
        import numpy as np
        f = np.array([100.0, 1000.0, 10000.0])
        lvl = np.array([120.0, 120.0, 120.0])
        assert np.allclose(apply_weighting(lvl, f, "LF"),
                           lvl + auditory_weighting(f, "LF"))
        assert np.isfinite(weighted_level(lvl, f, "LF"))
        # weighted_level integrates over frequency, so it must be independent
        # of the grid density (a bare sample-sum was not — it swung ~10 dB).
        f_coarse = np.linspace(20.0, 20000.0, 50)
        f_fine = np.linspace(20.0, 20000.0, 500)
        wl_coarse = weighted_level(np.full(f_coarse.size, 120.0), f_coarse, "LF")
        wl_fine = weighted_level(np.full(f_fine.size, 120.0), f_fine, "LF")
        assert abs(wl_coarse - wl_fine) < 0.5
        # Hand anchor: a flat 120 dB/Hz spectrum LF-weighted over 20 Hz-20 kHz
        # is 120 + 10·log10(∫ 10^(W/10) df) = 160.9719 dB re 1 µPa² on the
        # 500-point grid (the Southall closed form trapezoid-integrated).
        assert wl_fine == pytest.approx(160.9719, abs=1e-3)


class TestWindNoiseRollOffAnchor:
    """The >2 kHz roll-off must not depend on the caller's frequency grid.

    The melded high-frequency line is anchored at the 2 kHz cutoff itself,
    matching the rain roll-off and the no-sub-cutoff-sample fallback. Anchoring
    it at ``f_temp[-1]`` — the last in-grid sample below the cutoff — would
    move the 5 kHz level by 17.6 dB depending on whether the grid happens to
    contain 1999 Hz or 100 Hz.
    """

    @pytest.mark.parametrize('anchor', [10.0, 100.0, 500.0, 1500.0, 1999.0])
    def test_high_frequency_level_is_grid_independent(self, anchor):
        from uacpy.noise.noise import compute_windnoise
        probe = 5000.0
        with_anchor = compute_windnoise(np.array([anchor, probe]), 15.0)[-1]
        alone = compute_windnoise(np.array([probe]), 15.0)[-1]
        assert with_anchor == pytest.approx(alone, abs=1e-9), (
            f"grid containing {anchor} Hz shifts NL(5 kHz) by "
            f"{with_anchor - alone:.2f} dB")

    def test_curve_is_continuous_across_the_cutoff(self):
        from uacpy.noise.noise import compute_windnoise
        nl = compute_windnoise(np.array([1999.0, 2000.0, 2001.0]), 15.0)
        assert abs(nl[1] - nl[0]) < 0.05
        assert abs(nl[2] - nl[1]) < 0.05
