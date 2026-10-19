"""Tests for the ``uacpy.noise`` module.

``wind_noise_level`` and the ``WenzNoise`` class (wind / shipping / rain /
thermal / turbulence), ship-radiated noise, and the marine-mammal auditory
weightings — the levels themselves, and the three ways a caller can be misled
about them:

* **Unstated units.** ``WenzNoise`` took a wind speed in knots without saying
  so, unlike its ``chapman_harris_surface`` neighbour.
* **Silent substitution.** An unrecognised ``water_depth`` fell back to a
  default rather than saying it did not understand the argument.
* **NaN.** Every ``x < 0`` / ``x <= 0`` rejection here once let NaN through,
  because every comparison against NaN is False. The worst of them was
  ``WenzNoise(wind_speed_kn=nan)``: the NaN failed the wind model's
  ``blend > 0`` test, the wind component landed on the documented -inf
  switched-off sentinel, and ``.total`` came back all-finite and up to 63 dB
  below the true level with no warning. Each guard is now written as the
  negation of the admissible condition, with its legitimate zero pinned
  beside it.
"""

import inspect

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError, ValidityWarning
from uacpy.tests.conftest import recorded_warnings
from uacpy.noise import wind_noise_level, WenzNoise
from uacpy.noise import ambient as N

NAN = float('nan')
#: Three decades, three points — enough for a guard to reject or accept.
_GUARD_FREQS = np.array([100.0, 1000.0, 10000.0])
#: A resolved spectrum, for the tests that read levels rather than guards.
_WENZ_FREQS = np.logspace(1, 4, 40)
_DECIDECADE = 'decidecade'


@pytest.fixture
def freqs():
    # Log-spaced 1 Hz → 100 kHz so every Wenz band has plenty of points.
    return np.logspace(0.0, 5.0, 200)


# ─── wind_noise_level ───────────────────────────────────────────────────────


def test_compute_windnoise_zero_wind(freqs):
    """``u == 0`` silences the surface-noise source: spectral level is
    ``-inf`` dB at every frequency so the incoherent dB sum with the
    other Wenz components drops wind cleanly."""
    NL = wind_noise_level(freqs, wind_speed_kn=0)
    assert NL.shape == freqs.shape
    assert np.all(np.isneginf(NL))


def test_compute_windnoise_scalar_frequency():
    """``wind_noise_level(scalar_f, wind_speed_kn=u)`` returns a 1-element 1-D array."""
    NL = wind_noise_level(100.0, wind_speed_kn=10, water_depth='deep')
    assert NL.shape == (1,)
    assert np.isfinite(NL[0])


def test_compute_windnoise_negative_wind_raises():
    """Negative wind speed raises :class:`ConfigurationError`."""
    with pytest.raises(ConfigurationError, match="non-negative"):
        wind_noise_level(np.array([100.0]), wind_speed_kn=-5, water_depth='deep')


def test_compute_windnoise_takes_the_wind_speed_by_its_unit_name(freqs):
    """The wind speed is keyword-only and named for its unit, as in
    WenzNoise: a positional value states no unit, and a m/s reading taken as
    knots reads 5.7 dB low at 1 kHz."""
    with pytest.raises(TypeError, match='takes 1 positional argument but'):
        wind_noise_level(freqs, 10.0)
    with pytest.raises(TypeError, match="unexpected keyword argument 'u'"):
        wind_noise_level(freqs, u=10.0)
    np.testing.assert_array_equal(
        wind_noise_level(freqs, wind_speed_kn=10.0),
        WenzNoise(freqs, wind_speed_kn=10.0, wind_model='merklinger').wind)


def test_compute_windnoise_increases_with_wind(freqs):
    low = wind_noise_level(freqs, wind_speed_kn=5,  water_depth='deep')
    high = wind_noise_level(freqs, wind_speed_kn=25, water_depth='deep')
    assert np.all(np.isfinite(low))
    assert np.all(np.isfinite(high))
    band = (freqs >= 100) & (freqs <= 1500)
    assert np.mean(high[band]) > np.mean(low[band])


def test_compute_windnoise_shallow_louder_than_deep(freqs):
    deep = wind_noise_level(freqs, wind_speed_kn=10, water_depth='deep')
    shallow = wind_noise_level(freqs, wind_speed_kn=10, water_depth='shallow')
    band = (freqs >= 100) & (freqs <= 1500)
    assert np.mean(shallow[band]) > np.mean(deep[band])


def test_compute_windnoise_pinned_anchors():
    """Pin the wind-noise spectral level at four (wind, frequency) anchors.

    The values are DRDC-RDDC-2022-D051 eqs. (8)-(16) evaluated by hand for
    deep water (c0 = 42): they agree with ``wind_noise_level`` to ~1e-11 dB,
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
        got = wind_noise_level(np.array([fhz]), wind_speed_kn=u, water_depth='deep')
        assert got[0] == pytest.approx(level, abs=1e-6)


def test_wenznoise_high_freq_only_no_rain_meld_crash():
    """Rain melding only fires when both <7kHz and >7kHz frequencies exist."""
    freq_max = np.logspace(4.0, 5.0, 30)  # all > 7 kHz
    wenz = WenzNoise(freq_max, wind_speed_kn=10, rain_rate='moderate')
    assert np.all(np.isfinite(wenz.rain))


# ─── WenzNoise — invariants ─────────────────────────────────────────────────


def test_wenznoise_default_attributes(freqs):
    """Inactive bands are carried as ``-inf`` in each component array;
    ``.total`` stays finite as long as at least one source is active
    at every frequency."""
    wenz = WenzNoise(freqs, wind_speed_kn=10)
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
    default = WenzNoise(f, wind_speed_kn=10)
    explicit = WenzNoise(f, wind_speed_kn=10, shipping_level='medium',
                         rain_rate='no', water_depth='deep')
    np.testing.assert_array_equal(default.total, explicit.total)
    assert np.all(np.isneginf(default.rain))


def test_wenznoise_components_named(freqs):
    wenz = WenzNoise(freqs, wind_speed_kn=10, shipping_level='medium',
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
    wenz = WenzNoise(freqs, wind_speed_kn=10, shipping_level='medium',
                     rain_rate='moderate')
    components = np.column_stack([wenz.shipping, wenz.wind, wenz.rain,
                                  wenz.thermal, wenz.turbulence])
    # Total in dB must be ≥ each individual component everywhere.
    assert np.all(wenz.total + 1e-6 >= components.max(axis=1))


def test_wenznoise_shipping_levels_ordered(freqs):
    band = (freqs >= 30) & (freqs <= 200)
    low = WenzNoise(freqs, wind_speed_kn=5, shipping_level='low').total[band]
    med = WenzNoise(freqs, wind_speed_kn=5, shipping_level='medium').total[band]
    high = WenzNoise(freqs, wind_speed_kn=5, shipping_level='high').total[band]
    assert np.mean(low) < np.mean(med) < np.mean(high)


def test_wenznoise_as_psd_is_pa2_per_hz_of_the_db_levels(freqs):
    """The dB levels are re 1 µPa²/Hz and the linear PSD is SI Pa²/Hz:
    dividing by the 1 µPa reference squared returns ``total``."""
    import inspect
    wenz = WenzNoise(freqs, wind_speed_kn=10)
    pa2 = wenz.as_psd()
    np.testing.assert_allclose(10 * np.log10(pa2 / 1e-12), wenz.total,
                               rtol=0, atol=1e-9)
    # ``ref`` names only a dB reference, so a linear output takes none.
    assert list(inspect.signature(wenz.as_psd).parameters) == []


def test_wenznoise_band_level_integrates_the_spectrum_over_the_band():
    """The sonar-equation noise term for a band source level. It equals the
    trapezoid of ``as_psd`` on a dense grid of the same spectrum, does not
    depend on the grid the object was built on, tends to ``NL + 10·log10(w)``
    as the band narrows, and is the same number per component."""
    from uacpy.noise import WenzNoise
    coarse = WenzNoise(np.array([100.0]), wind_speed_kn=10.0)
    f = np.geomspace(500.0, 2000.0, 20001)
    dense = WenzNoise(f, wind_speed_kn=10.0)
    expected = 10.0 * np.log10(np.trapezoid(dense.as_psd(), f) / 1e-12)
    assert coarse.band_level(500.0, 2000.0) == pytest.approx(expected,
                                                             abs=1e-3)
    # The flat shortcut is off by the slope the band spans.
    flat = float(WenzNoise(np.array([1000.0]), wind_speed_kn=10.0).total[0])
    assert abs(flat + 10.0 * np.log10(1500.0) - expected) > 0.1
    # A narrow band is the spectral level times its width.
    narrow = coarse.band_level(999.5, 1000.5)
    assert narrow == pytest.approx(flat, abs=1e-3)
    wind = 10.0 * np.log10(np.trapezoid(10.0 ** (dense.wind / 10.0), f))
    assert coarse.band_level(500.0, 2000.0, component='wind') == \
        pytest.approx(wind, abs=1e-3)


@pytest.mark.parametrize('component', ['total', 'wind'])
def test_wenznoise_band_level_is_the_same_whatever_grid_it_was_built_on(
        component):
    """The band level reads the submodels, not the constructor grid: no
    grid, one bin, a dense grid and a grid outside the band give the same
    number, bit for bit."""
    from uacpy.noise import WenzNoise
    grids = [None, np.array([100.0]), np.geomspace(500.0, 2000.0, 2001),
             np.array([10.0, 20.0])]
    levels = [WenzNoise(g, wind_speed_kn=10.0).band_level(
        500.0, 2000.0, component=component) for g in grids]
    assert levels == [levels[0]] * len(grids)


@pytest.mark.parametrize('read', [
    lambda n: n.total, lambda n: n.wind, lambda n: n.shipping,
    lambda n: n.rain, lambda n: n.thermal, lambda n: n.turbulence,
    lambda n: n.components, lambda n: n.as_psd(), lambda n: n.plot(),
], ids=['total', 'wind', 'shipping', 'rain', 'thermal', 'turbulence',
        'components', 'as_psd', 'plot'])
def test_a_wenznoise_without_a_grid_refuses_its_spectrum_naming_frequencies(
        read):
    """Built with no ``frequencies=`` it holds band levels only; every
    spectrum view says which argument would give it one."""
    from uacpy.noise import WenzNoise
    noise = WenzNoise(wind_speed_kn=10.0)
    with pytest.raises(ConfigurationError, match='frequencies='):
        read(noise)


def test_the_band_level_grid_matches_adaptive_quadrature_of_the_model():
    """The 100-points-per-decade trapezoid of ``band_level`` against
    adaptive quadrature of the same submodels over four decades: measured
    4e-4 dB, pinned below 0.01 dB."""
    from scipy.integrate import quad
    from uacpy.core.acoustics import sum_levels_dB
    from uacpy.noise import WenzNoise
    from uacpy.noise.ambient import _eval_submodel
    noise = WenzNoise(wind_speed_kn=10.0)

    def density(f):
        f = np.atleast_1d(float(f))
        level = sum_levels_dB(*(
            _eval_submodel(noise._submodels[name], name, f, noise._params)
            for name in ('thermal', 'wind', 'shipping', 'turbulence', 'rain')))
        return float(10.0 ** (level[0] / 10.0))

    power, _ = quad(density, 10.0, 1e5, limit=500, epsrel=1e-10,
                    points=np.geomspace(10.0, 1e5, 30)[1:-1])
    assert noise.band_level(10.0, 1e5) == pytest.approx(
        10.0 * np.log10(power), abs=0.01)


def test_wenznoise_decidecade_levels_are_band_levels():
    from uacpy.acoustic_signal import decidecade_bands
    from uacpy.noise import WenzNoise
    noise = WenzNoise(np.array([100.0, 1000.0]), wind_speed_kn=10.0)
    centers, levels = noise.decidecade_levels(200.0, 2000.0)
    lower, expected_centers, upper = decidecade_bands(200.0, 2000.0)
    np.testing.assert_allclose(centers, expected_centers)
    np.testing.assert_allclose(
        levels, [noise.band_level(lo, hi) for lo, hi in zip(lower, upper)])


@pytest.mark.parametrize('kw, match', [
    (dict(freq_min=0.0, freq_max=100.0), '0 < freq_min < freq_max'),
    (dict(freq_min=200.0, freq_max=100.0), '0 < freq_min < freq_max'),
    (dict(freq_min=100.0, freq_max=200.0, component='ships'), 'component'),
])
def test_wenznoise_band_level_refuses_a_bad_band(kw, match):
    from uacpy.noise import WenzNoise
    noise = WenzNoise(np.array([100.0]), wind_speed_kn=10.0)
    with pytest.raises(ConfigurationError, match=match):
        noise.band_level(**kw)


def test_wenznoise_repr_contains_params(freqs):
    wenz = WenzNoise(freqs, wind_speed_kn=15, water_depth='shallow',
                     shipping_level='high', rain_rate='heavy')
    s = repr(wenz)
    assert s == ("WenzNoise(200 frequencies 1–100000 Hz, wind=15 kn, "
                 "depth=shallow, shipping=high, rain=heavy)")


def test_wenznoise_rejects_invalid_kwargs(freqs):
    with pytest.raises(ConfigurationError, match='water_depth'):
        WenzNoise(freqs, wind_speed_kn=10, water_depth='abyssal')
    with pytest.raises(ConfigurationError, match='shipping_level'):
        WenzNoise(freqs, wind_speed_kn=10, shipping_level='extreme')
    with pytest.raises(ConfigurationError, match='rain_rate'):
        WenzNoise(freqs, wind_speed_kn=10, rain_rate='monsoon')


def test_wenznoise_rejects_dc_and_negative_frequencies():
    # The empirical fits are all log10(f); a DC bin (common from a raw rfft
    # grid) would otherwise produce log10(0) = -inf/NaN, not a clear error.
    with pytest.raises(ConfigurationError, match='> 0 Hz'):
        WenzNoise(np.array([0.0, 10.0, 100.0]), wind_speed_kn=10)
    with pytest.raises(ConfigurationError, match='> 0 Hz'):
        WenzNoise(np.array([-5.0, 10.0]), wind_speed_kn=10)


def test_wenznoise_plot_returns_fig_ax(freqs):
    wenz = WenzNoise(freqs, wind_speed_kn=15)
    from uacpy.plot import plot_wenz
    fig, ax = plot_wenz(wenz)
    assert fig is not None and ax is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_wenznoise_plot_total_only(freqs):
    wenz = WenzNoise(freqs, wind_speed_kn=15)
    from uacpy.plot import plot_wenz
    fig, ax = plot_wenz(wenz, show_components=False)
    assert fig is not None and ax is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_wenznoise_total_matches_linear_sum_of_components(freqs):
    """``total`` equals the incoherent linear sum of the five
    components within numerical precision; ``-inf`` components
    contribute zero linear power."""
    wenz = WenzNoise(freqs, wind_speed_kn=10, shipping_level='medium',
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
    """``wind_speed_kn=0`` makes the wind component ``-inf`` so it drops out
    of ``.total``; the total equals the sum of the four remaining
    components."""
    wenz = WenzNoise(freqs, wind_speed_kn=0, shipping_level='medium',
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
    wenz = WenzNoise(f, wind_speed_kn=10.0, shipping_level='no')
    d = wenz.thermal - wenz.wind
    crossing = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
    assert crossing.size == 1
    assert 105e3 < f[crossing[0]] < 120e3
    f_calm = np.logspace(4.45, 4.6, 201)          # 28.2 - 39.8 kHz
    calm = WenzNoise(f_calm, wind_speed_kn=1.0, shipping_level='no')
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

    def test_rnl_uncertainty_dB_pinned(self):
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
    """Southall et al. 2019 auditory weighting applied to level spectra. The
    weighting itself, W(f), is tested with the other frequency weightings in
    test_acoustics_check_values.py."""

    def test_the_weighting_exported_here_is_the_core_one(self):
        """uacpy.noise exports the weighting from core.acoustics.weighting
        rather than a copy, so the spectrum helpers here and every
        weighting= argument read one parameter table."""
        import uacpy.noise as noise
        from uacpy.core.acoustics import weighting as core
        for name in ('auditory_weighting', 'WEIGHTING_PARAMS',
                     'HEARING_GROUPS', 'EXPOSURE_THRESHOLDS'):
            assert getattr(noise, name) is getattr(core, name)

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
        wenz = WenzNoise(f, wind_speed_kn=10.0, shipping_level='medium')
        unweighted = 10.0 * np.log10(np.trapezoid(10.0 ** (wenz.total / 10.0), f))
        assert unweighted == pytest.approx(97.5772, abs=5e-3)
        assert weighted_level(wenz.total, frequency=f, weighting="LF") == pytest.approx(94.0292, abs=5e-3)
        assert weighted_level(wenz.total, frequency=f, weighting="VHF") == pytest.approx(83.5095, abs=5e-3)

    def test_apply_and_weighted_level(self):
        from uacpy.noise import apply_weighting, weighted_level, auditory_weighting
        import numpy as np
        f = np.array([100.0, 1000.0, 10000.0])
        lvl = np.array([120.0, 120.0, 120.0])
        assert np.allclose(apply_weighting(lvl, frequency=f, weighting="LF"),
                           lvl + auditory_weighting(f, "LF"))
        assert np.isfinite(weighted_level(lvl, frequency=f, weighting="LF"))
        # weighted_level integrates over frequency, so it must be independent
        # of the grid density (a bare sample-sum was not — it swung ~10 dB).
        f_coarse = np.linspace(20.0, 20000.0, 50)
        f_fine = np.linspace(20.0, 20000.0, 500)
        wl_coarse = weighted_level(np.full(f_coarse.size, 120.0), frequency=f_coarse, weighting="LF")
        wl_fine = weighted_level(np.full(f_fine.size, 120.0), frequency=f_fine, weighting="LF")
        assert abs(wl_coarse - wl_fine) < 0.5
        # Hand anchor: a flat 120 dB/Hz spectrum LF-weighted over 20 Hz-20 kHz
        # is 120 + 10·log10(∫ 10^(W/10) df) = 160.9719 dB re 1 µPa² on the
        # 500-point grid (the Southall closed form trapezoid-integrated).
        assert wl_fine == pytest.approx(160.9719, abs=1e-3)

    def test_apply_weighting_weighs_along_the_axis_named(self):
        """A (frequency, depth) map is weighted along axis 0 with axis=0,
        as weighted_level reads it; the default is the last axis, and a
        length that matches no frequency is a typed refusal, not numpy's
        broadcast error."""
        from uacpy.noise import apply_weighting, auditory_weighting
        f = np.array([100.0, 1000.0, 10000.0])
        w = auditory_weighting(f, 'VHF')
        square = np.full((3, 3), 100.0)
        np.testing.assert_allclose(
            apply_weighting(square, frequency=f, weighting='VHF', axis=0),
            np.broadcast_to(100.0 + w[:, None], (3, 3)))
        np.testing.assert_allclose(
            apply_weighting(square, frequency=f, weighting='VHF'),
            np.broadcast_to(100.0 + w[None, :], (3, 3)))
        with pytest.raises(ConfigurationError, match='one level per frequency'):
            apply_weighting(np.full((3, 5), 100.0), frequency=f,
                            weighting='VHF')
        assert apply_weighting(np.full((3, 5), 100.0), frequency=f,
                               weighting='VHF', axis=0).shape == (3, 5)

    @pytest.mark.parametrize('group', ['LF', 'VHF'])
    def test_a_silent_spectrum_weighs_to_the_marker(self, group):
        """Every bin at the no-energy marker (a silent record's welch
        level): the weighting does not move a marker bin, so the weighted
        level is the marker, not -600 + W + 10·log10(bandwidth) (-559.3 dB
        LF-weighted over 1-20 kHz, which the mask reads as a level)."""
        from uacpy.acoustics import power_to_dB
        from uacpy.noise import weighted_level
        f = np.linspace(1000.0, 20000.0, 200)
        assert weighted_level(power_to_dB(np.zeros(200)), frequency=f,
                              weighting=group) == -600.0

    def test_weighted_level_refuses_fewer_than_two_frequencies(self):
        """The integral over frequency spans no bandwidth on fewer than two
        samples, so a scalar and a one-element spectrum both raise the typed
        error naming ``apply_weighting`` as the single-frequency form — not a
        bare IndexError (the scalar) or a silent
        10·log10(float-tiny) = -3076.5 dB (the one-element array)."""
        from uacpy.noise import apply_weighting, weighted_level
        with pytest.raises(ConfigurationError, match='apply_weighting') as err:
            weighted_level(120.0, frequency=1000.0, weighting="LF")
        # The remedy, typed back, runs (keyword-only arguments).
        assert 'apply_weighting(psd_dB, frequency=f, weighting=weighting)' in str(
            err.value)
        assert np.isfinite(apply_weighting(120.0, frequency=1000.0, weighting="LF"))
        with pytest.raises(ConfigurationError, match='at least two'):
            weighted_level(np.array([120.0]), frequency=np.array([1000.0]), weighting="LF")

    def test_weighted_level_accepts_the_two_frequency_boundary(self):
        """n = 2 — a single trapezoid interval — is the smallest valid grid:
        flat 120 dB/Hz LF-weighted over 1-2 kHz integrates to
        149.9634 dB re 1 µPa² (120 + W(f) trapezoid-integrated by hand)."""
        from uacpy.noise import weighted_level
        f = np.array([1000.0, 2000.0])
        lvl = np.array([120.0, 120.0])
        assert weighted_level(lvl, frequency=f, weighting="LF") == pytest.approx(
            149.9634445674845, abs=1e-9)


class TestTheWeightedLevelTakesTheSpectraUacpyMakes:
    """``weighted_level`` / ``apply_weighting`` take ``weighting=`` through
    ``frequency_weighting``: a welch spectrum with its 0 Hz bin goes in as it
    comes (no weight at DC), a function or a table weighs as a group does,
    and an N-D density gives one level per spectrum along ``axis``."""

    def test_a_welch_spectrum_with_its_dc_bin_is_weighted_as_it_comes(self):
        from uacpy.acoustic_signal import welch
        from uacpy.acoustics import power_to_dB
        from uacpy.noise import weighted_level
        rng = np.random.default_rng(0)
        f, psd = welch(rng.standard_normal(48000), 48000.0, nperseg=1024)
        assert f[0] == 0.0
        level = power_to_dB(psd)
        with_dc = weighted_level(level, frequency=f, weighting='HF')
        assert np.isfinite(with_dc)
        assert with_dc == pytest.approx(
            weighted_level(level[1:], frequency=f[1:], weighting='HF'),
            abs=1e-9)

    def test_a_function_and_a_table_weigh_as_the_group_does(self):
        from uacpy.noise import auditory_weighting, weighted_level
        f = np.geomspace(100.0, 20000.0, 400)
        lvl = np.full(f.size, 100.0)
        group = weighted_level(lvl, frequency=f, weighting='LF')
        fn = weighted_level(lvl, frequency=f,
                            weighting=lambda x: auditory_weighting(x, 'LF'))
        table = weighted_level(lvl, frequency=f,
                               weighting=(f, auditory_weighting(f, 'LF')))
        assert fn == pytest.approx(group, abs=1e-12)
        assert table == pytest.approx(group, abs=1e-9)

    def test_an_nd_density_gives_one_level_per_spectrum(self):
        from uacpy.noise import weighted_level
        f = np.geomspace(100.0, 20000.0, 200)
        lvl = np.stack([np.full(f.size, 100.0), np.full(f.size, 110.0)])
        got = weighted_level(lvl, frequency=f, weighting='VHF')
        assert got.shape == (2,)
        assert got[1] - got[0] == pytest.approx(10.0, abs=1e-9)
        np.testing.assert_allclose(
            weighted_level(lvl.T, frequency=f, weighting='VHF', axis=0), got,
            rtol=1e-13)


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
        from uacpy.noise.ambient import wind_noise_level
        probe = 5000.0
        with_anchor = wind_noise_level(np.array([anchor, probe]), wind_speed_kn=15.0)[-1]
        alone = wind_noise_level(np.array([probe]), wind_speed_kn=15.0)[-1]
        assert with_anchor == pytest.approx(alone, abs=1e-9), (
            f"grid containing {anchor} Hz shifts NL(5 kHz) by "
            f"{with_anchor - alone:.2f} dB")

    def test_curve_is_continuous_across_the_cutoff(self):
        from uacpy.noise.ambient import wind_noise_level
        nl = wind_noise_level(np.array([1999.0, 2000.0, 2001.0]), wind_speed_kn=15.0)
        assert abs(nl[1] - nl[0]) < 0.05
        assert abs(nl[2] - nl[1]) < 0.05


class TestWenzNoiseRefusesNonFiniteWind:
    """The headline: a NaN wind speed used to return the switched-off spectrum."""

    def test_nan_wind_speed_raises_instead_of_returning_the_zero_wind_spectrum(self):
        with pytest.raises(
                ConfigurationError,
                match="wind_speed_kn must be non-negative"):
            WenzNoise(_GUARD_FREQS, wind_speed_kn=NAN)

    def test_zero_wind_gives_the_switched_off_wind_component(self):
        # u == 0 is the documented sentinel and must keep working: it is what
        # the NaN was silently impersonating.
        assert np.all(np.isfinite(np.asarray(WenzNoise(_GUARD_FREQS, wind_speed_kn=0.0).total)))

    def test_the_zero_wind_spectrum_it_impersonated_is_tens_of_dB_low(self):
        # Pre-fix, WenzNoise(wind_speed_kn=nan).total equalled the wind=0 spectrum
        # exactly; this pins the size of the error that hid behind that.
        f = np.logspace(1, 5, 200)
        deficit = (np.asarray(WenzNoise(f, wind_speed_kn=10.0).total)
                   - np.asarray(WenzNoise(f, wind_speed_kn=0.0).total))
        assert deficit.max() > 40.0

    def test_nan_frequency_bin_raises(self):
        with pytest.raises(ConfigurationError, match="frequencies must be > 0 Hz and finite"):
            WenzNoise(np.array([NAN, 1000.0]), wind_speed_kn=5.0)

    def test_negative_wind_speed_raises(self):
        with pytest.raises(ConfigurationError, match="wind_speed_kn"):
            WenzNoise(_GUARD_FREQS, wind_speed_kn=-1.0)


class TestNoisePositivityGuardsRefuseNaN:
    def test_compute_windnoise_nan_wind_raises(self):
        with pytest.raises(ConfigurationError, match=r"non-negative \(knots\) and finite"):
            wind_noise_level(_GUARD_FREQS, wind_speed_kn=NAN)

    def test_compute_windnoise_dc_bin_raises(self):
        with pytest.raises(ConfigurationError, match="frequencies must be > 0 Hz and finite"):
            wind_noise_level(np.array([0.0, 100.0]), wind_speed_kn=10.0)

    def test_compute_windnoise_nan_frequency_bin_raises(self):
        with pytest.raises(ConfigurationError, match="frequencies must be > 0 Hz and finite"):
            wind_noise_level(np.array([NAN, 100.0]), wind_speed_kn=10.0)

    def test_compute_windnoise_zero_wind_returns_minus_inf(self):
        assert np.all(np.isneginf(wind_noise_level(_GUARD_FREQS, wind_speed_kn=0.0)))

class TestWenzNoiseNamesItsWindSpeedUnit:
    """``chapman_harris_surface(wind_speed_kn=)`` and ``WenzNoise`` take the
    same unit, knots, and the name says so: the unit the coefficients are
    natively in (DRDC-RDDC-2022-D051 §2.3 eq. 8). Feeding a 10 m/s reading
    in as knots understates the total by 5.74 dB at 1 kHz."""

    def test_the_parameter_is_named_for_its_unit(self):
        params = inspect.signature(N.WenzNoise.__init__).parameters
        assert 'wind_speed_kn' in params
        assert 'wind_speed' not in params

    def test_the_parameter_is_keyword_only(self):
        """A positional ``WenzNoise(f, 10)`` states no unit at all, so it would
        have survived the rename untouched."""
        assert (inspect.signature(N.WenzNoise.__init__)
                .parameters['wind_speed_kn'].kind
                is inspect.Parameter.KEYWORD_ONLY)
        with pytest.raises(TypeError,
                           match='positional arguments but 3 were given'):
            N.WenzNoise(_WENZ_FREQS, 10.0)

    def test_the_attribute_carries_the_unit_too(self):
        w = N.WenzNoise(_WENZ_FREQS, wind_speed_kn=10.0)
        assert w.wind_speed_kn == 10.0
        assert not hasattr(w, 'wind_speed')
        assert '10 kn' in repr(w)

    def test_the_submodel_protocol_keyword_carries_the_unit(self):
        """Third-party submodels registered with
        ``register_noise_model('wind', 'mine', fn)``
        receive the bundle by keyword, so the protocol name has to move with
        the constructor's or a custom model silently misses its wind."""
        seen = {}

        def probe(frequencies, *, wind_speed_kn, **_):
            seen['kn'] = wind_speed_kn
            return np.full_like(frequencies, 50.0)

        N.WenzNoise(_WENZ_FREQS, wind_speed_kn=12.5, wind_model=probe)
        assert seen['kn'] == 12.5
        for name in ('merklinger', 'coates'):
            sig = inspect.signature(N.WIND_MODELS[name])
            assert 'wind_speed_kn' in sig.parameters
            assert 'wind_speed' not in sig.parameters

    def test_a_metres_per_second_reading_read_as_knots_is_quiet_not_equal(self):
        """The error the name prevents, measured: 10 m/s is 19.44 kn."""
        f = np.array([100.0, 1000.0, 10000.0])
        wrong = N.WenzNoise(f, wind_speed_kn=10.0).total
        right = N.WenzNoise(f, wind_speed_kn=10.0 * 1.9438445).total
        deficit = right - wrong
        assert deficit == pytest.approx([0.803, 5.740, 5.768], abs=0.01)


class TestUnrecognisedWaterDepthRaisesInsteadOfSubstituting:
    """``WenzNoise.__init__`` already raised on this parameter with this value
    set; the two module-level entry points did not. Worse, they disagreed:
    through ``WIND_MODELS['merklinger']`` an unknown string meant deep, and
    through ``SHIPPING_MODELS['wenz']`` the same string meant shallow."""

    @pytest.mark.parametrize('bad', ['SHALLOW', 'Shallow', 'shallow ',
                                     'abyssal', 'medium'])
    def test_compute_windnoise_refuses_it(self, bad):
        with pytest.raises(ConfigurationError, match="'deep' or 'shallow'"):
            N.wind_noise_level(_WENZ_FREQS, wind_speed_kn=10.0, water_depth=bad)

    @pytest.mark.parametrize('bad', ['SHALLOW', 'Shallow', 'abyssal'])
    def test_the_wenz_shipping_submodel_refuses_it(self, bad):
        with pytest.raises(ConfigurationError, match="'deep' or 'shallow'"):
            N.SHIPPING_MODELS['wenz'](_WENZ_FREQS, shipping_level='medium',
                                      water_depth=bad)

    def test_a_silenced_source_is_no_excuse_to_skip_the_check(self):
        """Both submodels short-circuit on a switched-off source — ``u == 0``
        for wind, ``shipping_level='no'`` for shipping — before they ever read
        the coefficient family. The guard sits ahead of both short circuits so
        the parameter is refused on every path, not only the ones that reach
        the fit."""
        with pytest.raises(ConfigurationError, match="'deep' or 'shallow'"):
            N.wind_noise_level(_WENZ_FREQS, wind_speed_kn=0.0, water_depth='SHALLOW')
        with pytest.raises(ConfigurationError, match="'deep' or 'shallow'"):
            N.SHIPPING_MODELS['wenz'](_WENZ_FREQS, shipping_level='no',
                                      water_depth='SHALLOW')

    @pytest.mark.parametrize('depth', ['deep', 'shallow'])
    def test_both_recognised_families_compute_finite_levels(self, depth):
        wind = N.wind_noise_level(_WENZ_FREQS, wind_speed_kn=10.0, water_depth=depth)
        ship = N.SHIPPING_MODELS['wenz'](_WENZ_FREQS, shipping_level='medium',
                                         water_depth=depth)
        assert np.all(np.isfinite(wind))
        assert np.all(np.isfinite(ship))

    def test_an_omitted_water_depth_defaults_to_deep(self):
        """The default for an *absent* argument is what makes it wrong for an
        unrecognised one, so it has to stay."""
        assert np.allclose(N.wind_noise_level(_WENZ_FREQS, wind_speed_kn=10.0),
                           N.wind_noise_level(_WENZ_FREQS, wind_speed_kn=10.0, water_depth='deep'))

    def test_the_two_families_are_far_enough_apart_to_matter(self):
        """3.0 dB of wind at 50 Hz, and the shipping hump moves 30 Hz → 65 Hz.
        Both were silently reachable by a typo."""
        f = np.array([50.0])
        deep = N.wind_noise_level(f, wind_speed_kn=10.0, water_depth='deep')[0]
        shallow = N.wind_noise_level(f, wind_speed_kn=10.0, water_depth='shallow')[0]
        assert shallow - deep == pytest.approx(3.0, abs=0.01)


class TestWindNoiseModelsAgreeAtZeroWind:
    """``wind_noise_level`` returns -inf at u = 0 so a switched-off source
    contributes nothing to the incoherent dB sum. The Coates wind model has no
    term that vanishes with the wind, so without an explicit zero case it
    returned ~44 dB re 1 µPa²/Hz at 1 kHz in a flat calm and raised the Wenz
    total by 14.7 dB at 1 kHz / 24.3 dB at 10 kHz."""

    F = np.array([100.0, 1000.0, 10000.0])

    def test_coates_wind_is_silent_at_zero_wind(self):
        assert np.all(np.isneginf(N.WIND_MODELS['coates'](self.F,
                                                          wind_speed_kn=0.0)))

    def test_totals_match_the_default_model_at_zero_wind(self):
        default = N.WenzNoise(self.F, wind_speed_kn=0.0)
        coates = N.WenzNoise(self.F, wind_speed_kn=0.0, wind_model='coates')
        assert np.all(np.isneginf(coates.wind))
        np.testing.assert_allclose(coates.total, default.total)

    def test_a_nineteen_knot_coates_wind_lands_between_60_and_75_dB(self):
        w = N.WIND_MODELS['coates'](np.array([1000.0]), wind_speed_kn=19.4)[0]
        assert 60.0 < w < 75.0


class TestSilentSubmodelsReturnFloats:
    """``np.full_like`` on an integer frequency vector casts -inf to
    INT64_MIN, and the submodel registries are public entry points."""

    FI = np.array([100, 1000, 10000])          # integer dtype on purpose

    @pytest.mark.parametrize("model, kwargs", [
        (N.SHIPPING_MODELS['wenz'], dict(shipping_level='no',
                                         water_depth='deep')),
        (N.SHIPPING_MODELS['coates'], dict(shipping_level='no')),
        (N.RAIN_MODELS['torres_costa'], dict(rain_rate='no')),
        (N.WIND_MODELS['coates'], dict(wind_speed_kn=0.0)),
    ])
    def test_silent_source_is_float_minus_inf(self, model, kwargs):
        out = model(self.FI, **kwargs)
        assert out.dtype == np.float64
        assert np.all(np.isneginf(out))


class TestShipRadiatedNoiseGuardsRefuseNaN:
    """Every comparison against NaN is False, so ``r <= 0`` / ``draught <= 0``
    let a NaN through to a NaN level, and ``lloyd_mirror_correction`` had no
    depth guard at all — ``kd`` enters only as ``(kd)**2`` and ``(kd)**4``, so
    a negative depth returns bit-identically what its positive twin returns
    and an upstream sign error is invisible."""

    FREQ = np.array([100.0, 1000.0])

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), -5.0, 0.0])
    def test_radiated_noise_level_refuses_a_bad_distance(self, bad):
        from uacpy.noise.ship_radiated_noise import radiated_noise_level
        with pytest.raises(ConfigurationError, match='distance'):
            radiated_noise_level(120.0, bad)

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), -3.0, 0.0])
    def test_nominal_source_depth_refuses_a_bad_draught(self, bad):
        from uacpy.noise.ship_radiated_noise import nominal_source_depth
        with pytest.raises(ConfigurationError, match='draught'):
            nominal_source_depth(bad)

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), -5.0])
    def test_lloyd_mirror_refuses_a_negative_or_non_finite_depth(self, bad):
        from uacpy.noise.ship_radiated_noise import lloyd_mirror_correction
        with pytest.raises(ConfigurationError, match='source_depth'):
            lloyd_mirror_correction(self.FREQ, bad)

    def test_a_fleet_of_draughts_and_depths_is_evaluated_elementwise(self):
        from uacpy.noise.ship_radiated_noise import (lloyd_mirror_correction,
                                                     nominal_source_depth)
        draughts = np.array([4.0, 8.0, 12.0])
        depths = nominal_source_depth(draughts)
        np.testing.assert_array_equal(
            depths, [nominal_source_depth(d) for d in draughts])
        assert isinstance(nominal_source_depth(8.0), float)
        f = np.array([50.0, 200.0, 1000.0])
        np.testing.assert_array_equal(
            lloyd_mirror_correction(f, depths),
            [lloyd_mirror_correction(fi, di) for fi, di in zip(f, depths)])
        with pytest.raises(ConfigurationError, match='draught'):
            nominal_source_depth(np.array([4.0, -1.0]))
        with pytest.raises(ConfigurationError, match='source_depth'):
            lloyd_mirror_correction(f, np.array([1.0, np.nan, 2.0]))

    def test_zero_depth_is_the_admissible_boundary(self):
        """Both sides of the depth guard: ``kd -> 0`` is the physical
        surface-mounted limit the function's own ``errstate`` is written for,
        so it stays admissible and returns ``+inf``."""
        from uacpy.noise.ship_radiated_noise import lloyd_mirror_correction
        with np.errstate(divide='ignore'):
            at_zero = lloyd_mirror_correction(self.FREQ, 0.0)
        assert np.all(np.isposinf(at_zero))
        with pytest.raises(ConfigurationError,
                           match='source_depth must be >= 0 m and finite'):
            lloyd_mirror_correction(self.FREQ, -1e-12)

    def test_a_positive_depth_returns_the_iso_correction(self):
        from uacpy.noise.ship_radiated_noise import lloyd_mirror_correction
        got = lloyd_mirror_correction(self.FREQ, 5.0)
        assert np.all(np.isfinite(got))
        assert got[-1] == pytest.approx(-3.059, abs=1e-3)

    @pytest.mark.parametrize('bad', [float('nan'), 0.0, -1500.0])
    def test_lloyd_mirror_refuses_a_bad_sound_speed(self, bad):
        from uacpy.noise.ship_radiated_noise import lloyd_mirror_correction
        with pytest.raises(ConfigurationError, match='sound_speed'):
            lloyd_mirror_correction(self.FREQ, 5.0, bad)


# ── one band integrator; an empty band is -inf everywhere ─────────────────


def test_every_band_integrator_reads_an_empty_band_as_minus_inf():
    """band_levels, WenzNoise.band_level and weighted_level all
    integrate through
    core.acoustics.band_level, and a band that carries no power is -inf in
    each, neither a floored level nor NaN."""
    from uacpy.acoustic_signal import band_levels
    from uacpy.noise import weighted_level
    f = np.fft.rfftfreq(4096, 1 / 8000.0)
    psd = np.zeros(f.size)
    psd[(f > 900) & (f < 1100)] = 1e-4
    _, levels = band_levels(psd, frequencies=f)
    covered = ~np.isnan(levels)
    assert np.isneginf(levels[covered]).any()
    assert np.isfinite(levels[covered]).any()
    silent = WenzNoise(np.array([100.0]), wind_speed_kn=0.0,
                       shipping_level='no', rain_rate='no')
    assert silent.band_level(100.0, 1000.0, component='wind') == -np.inf
    fw = np.geomspace(10.0, 1e5, 50)
    assert weighted_level(np.full(fw.size, -np.inf), frequency=fw, weighting='LF') == -np.inf


def test_wenz_decidecade_levels_carry_their_bands_and_reference():
    """``WenzNoise.decidecade_levels`` returns a ``BandLevels`` on the
    decidecade ladder in dB re 1 µPa², its edges those of
    ``decidecade_bands`` and each level the ``band_level`` of its band."""
    from uacpy.acoustic_signal import BandLevels, decidecade_bands
    from uacpy.noise import WenzNoise
    w = WenzNoise(np.logspace(1, 4, 400), wind_speed_kn=10.0)
    result = w.decidecade_levels(100.0, 1000.0)
    assert isinstance(result, BandLevels)
    lower, centres, upper = decidecade_bands(100.0, 1000.0)
    np.testing.assert_array_equal(result.centers, centres)
    np.testing.assert_array_equal(result.lower, lower)
    np.testing.assert_array_equal(result.upper, upper)
    assert (result.band_type, result.ref) == ('decidecade', 1e-6)
    assert result.levels_dB[3] == w.band_level(lower[3], upper[3])


class TestTheMeanShipSourceSpectrumIsWalesAndHeitmeyer:
    """``ship_source_level_wales_heitmeyer`` is the Wales and Heitmeyer
    (2002) mean monopole spectrum as Ainslie (2010) Eq. (8.195) gives it,
    ``230.0 - 35.94·log10 f + 9.17·log10[1 + (f/340)²]`` dB re 1 µPa² m²/Hz
    over the measured 30-1200 Hz, then sect. 8.3.1.4's -23 dB/decade
    extrapolation, joined at 1200 Hz, to 15 kHz. The values are the equation
    evaluated by hand."""

    @pytest.mark.parametrize('f, level', [
        (30.0, 176.9431), (100.0, 158.4504), (340.0, 141.7791),
        (1000.0, 131.2083), (1200.0, 129.6866),
        (2000.0, 124.5841), (10000.0, 108.5078), (15000.0, 104.4577)])
    def test_it_is_equation_8_195_and_its_extrapolation(self, f, level):
        from uacpy.noise import ship_source_level_wales_heitmeyer
        assert ship_source_level_wales_heitmeyer(f) == pytest.approx(
            level, abs=1e-4)

    def test_the_extrapolation_falls_23_dB_per_decade_from_the_join(self):
        from uacpy.noise import ship_source_level_wales_heitmeyer
        levels = ship_source_level_wales_heitmeyer(
            np.array([1200.0, 1200.0 * (1 + 1e-9), 1500.0, 15000.0]))
        assert levels[1] == pytest.approx(levels[0], abs=1e-6)
        assert levels[2] - levels[3] == pytest.approx(23.0, abs=1e-9)

    def test_an_empty_frequency_is_refused(self):
        from uacpy.noise import ship_source_level_wales_heitmeyer
        with pytest.raises(ConfigurationError,
                           match=r'^ship_source_level_wales_heitmeyer: '
                                 r'frequency is empty\.'):
            ship_source_level_wales_heitmeyer([])

    @pytest.mark.parametrize('f, outside', [(30.0, False), (29.9, True),
                                            (15000.0, False),
                                            (15000.1, True)])
    def test_outside_the_measured_and_extrapolated_band_it_warns(self, f,
                                                                 outside):
        """Outside 30 Hz-15 kHz the expressions are evaluated with one
        ValidityWarning naming both bands; inside, silently."""
        from uacpy.noise import ship_source_level_wales_heitmeyer
        with recorded_warnings() as caught:
            level = ship_source_level_wales_heitmeyer(f)
        named = [w for w in caught if issubclass(w.category, ValidityWarning)
                 and 'measured over 30-1200 Hz' in str(w.message)
                 and 'extrapolated to 15000 Hz' in str(w.message)]
        assert len(named) == (1 if outside else 0)
        assert np.isfinite(level)

    def test_below_30_Hz_it_is_equation_8_195_and_above_15_kHz_the_line(self):
        from uacpy.noise import ship_source_level_wales_heitmeyer
        with recorded_warnings():
            low, high = ship_source_level_wales_heitmeyer([20.0, 20000.0])
        assert low == pytest.approx(
            230.0 - 35.94 * np.log10(20.0)
            + 9.17 * np.log10(1.0 + (20.0 / 340.0) ** 2), abs=1e-9)
        assert high == pytest.approx(
            ship_source_level_wales_heitmeyer(15000.0)
            - 23.0 * np.log10(20000.0 / 15000.0), abs=1e-9)

    @pytest.mark.parametrize('f', [0.0, -10.0, np.nan, np.inf])
    def test_a_frequency_without_a_logarithm_is_refused(self, f):
        from uacpy.noise import ship_source_level_wales_heitmeyer
        with pytest.raises(ConfigurationError,
                           match=r'^ship_source_level_wales_heitmeyer: '
                                 r'frequency must be finite and positive'):
            ship_source_level_wales_heitmeyer([100.0, f])
