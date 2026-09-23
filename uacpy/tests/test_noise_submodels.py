"""Tests for the pluggable submodel registries behind ``WenzNoise``.

The expected side of each formula test is a hand transcription of the DRDC
report's NUMBERED equations, with the equation number named in the docstring —
not a call back into ``uacpy.noise``. That is what the file is for: uacpy
implements the numbered equations, and the report's Annex A Matlab listing
disagrees with them in two places (the wind melding exponent, and the >2 kHz
anchor). Those two disagreements are the subject of
:func:`test_wind_follows_drdc_annex_a_below_the_cutoff` and
:func:`test_wind_above_the_cutoff_does_not_depend_on_the_frequency_grid`, so a
reader hitting a mismatch should re-read the report before changing either side.
"""

import numpy as np
import pytest
from uacpy.noise import ambient as N
from uacpy.core.exceptions import ConfigurationError

F = np.array([10.0, 100.0, 1000.0, 10000.0])


def test_registries_have_defaults():
    assert 'merklinger' in N.WIND_MODELS
    assert 'wenz' in N.SHIPPING_MODELS
    assert 'torres_costa' in N.RAIN_MODELS
    assert 'mellen' in N.THERMAL_MODELS
    assert 'wenz' in N.TURBULENCE_MODELS


def test_thermal_matches_formula():
    # No 0-dB floor: the Mellen thermal PSD (4πKT(ρ/c)f², Abraham eq. 3.193) is
    # strictly positive, so its level is legitimately negative below ~5.6 kHz.
    out = N.THERMAL_MODELS['mellen'](F)
    ref = -75.0 + 20.0 * np.log10(F)
    assert np.array_equal(out, ref)


def test_turbulence_follows_drdc():
    """DRDC §2.1: ``NL_turb = NL_t + m_t*log10(f_Hz)`` with ``NL_t = 107 dB``
    and ``m_t = -10 dB/octave`` (= -33.2 dB/decade). Real (possibly sub-0-dB)
    levels are kept, so the curve is genuinely negative at high frequency."""
    out = N.TURBULENCE_MODELS['wenz'](F)
    ref = 107.0 - (10.0 / np.log10(2.0)) * np.log10(F)
    assert np.allclose(out, ref, equal_nan=True)
    # NL_t: the level at 1 Hz.
    assert np.isclose(N.TURBULENCE_MODELS['wenz'](np.array([1.0]))[0], 107.0)
    # The slope is specified per octave; a decade is 1/log10(2) octaves.
    decade = np.array([10.0, 100.0])
    per_decade = np.diff(N.TURBULENCE_MODELS['wenz'](decade))[0]
    assert np.isclose(per_decade * np.log10(2.0), -10.0)


def test_shipping_no_is_silent():
    out = N.SHIPPING_MODELS['wenz'](F, shipping_level='no', water_depth='deep')
    assert np.all(out == -np.inf)


def test_resolve_submodel():
    fn, name = N._resolve_submodel(None, N.WIND_MODELS, 'merklinger', 'wind_model')
    assert name == 'merklinger' and callable(fn)
    fn, name = N._resolve_submodel('merklinger', N.WIND_MODELS, 'merklinger', 'wind_model')
    assert name == 'merklinger'

    def custom(f, **k):
        return np.zeros_like(f)
    fn, name = N._resolve_submodel(custom, N.WIND_MODELS, 'merklinger', 'wind_model')
    assert name == 'custom' and fn is custom
    with pytest.raises(ConfigurationError):
        N._resolve_submodel('nope', N.WIND_MODELS, 'merklinger', 'wind_model')
    with pytest.raises(ConfigurationError):
        N._resolve_submodel(123, N.WIND_MODELS, 'merklinger', 'wind_model')


def test_default_matches_registry_defaults():
    w = N.WenzNoise(F, wind_speed_kn=15.0, rain_rate='heavy',
                    water_depth='deep', shipping_level='high')
    th = N.THERMAL_MODELS['mellen'](F)
    wi = N.WIND_MODELS['merklinger'](F, wind_speed_kn=15.0, water_depth='deep')
    sh = N.SHIPPING_MODELS['wenz'](F, shipping_level='high', water_depth='deep')
    tu = N.TURBULENCE_MODELS['wenz'](F)
    ra = N.RAIN_MODELS['torres_costa'](F, rain_rate='heavy')
    ln10 = np.log(10.0)
    total = (10.0 / ln10) * np.logaddexp.reduce(
        np.stack([th, wi, sh, tu, ra]) * (ln10 / 10.0), axis=0)
    assert np.array_equal(w.thermal, th) and np.array_equal(w.wind, wi)
    assert np.array_equal(w.shipping, sh) and np.array_equal(w.turbulence, tu)
    assert np.array_equal(w.rain, ra)
    assert np.allclose(w.total, total, rtol=0, atol=0, equal_nan=True)


def test_models_recorded():
    w = N.WenzNoise(F, wind_speed_kn=15.0)
    assert w.models == {'wind': 'merklinger', 'shipping': 'wenz',
                        'rain': 'torres_costa', 'thermal': 'mellen',
                        'turbulence': 'wenz'}


def test_string_selector_equals_default():
    a = N.WenzNoise(F, wind_speed_kn=15.0)
    b = N.WenzNoise(F, wind_speed_kn=15.0, wind_model='merklinger')
    assert np.array_equal(a.wind, b.wind)


def test_bad_selector_raises():
    with pytest.raises(ConfigurationError):
        N.WenzNoise(F, wind_speed_kn=15.0, wind_model='nope')


def test_custom_callable_changes_only_that_component():
    base = N.WenzNoise(F, wind_speed_kn=15.0)
    flat = N.WenzNoise(F, wind_speed_kn=15.0, wind_model=lambda f, **k: np.full_like(f, 50.0))
    assert np.allclose(flat.wind, 50.0)
    assert np.array_equal(flat.shipping, base.shipping)
    assert flat.models['wind'] == 'custom'


def test_components_namedtuple():
    w = N.WenzNoise(F, wind_speed_kn=15.0, rain_rate='light')
    c = w.components
    assert isinstance(c, N.NoiseComponents)
    assert c._fields == ('total', 'wind', 'shipping', 'rain',
                         'thermal', 'turbulence')
    assert np.array_equal(c.wind, w.wind)
    assert np.array_equal(c.total, w.total)


def test_registry_extensible_and_exported():
    import uacpy.noise as pkg
    pkg.WIND_MODELS['flat50'] = lambda f, **k: np.full_like(f, 50.0)
    try:
        w = N.WenzNoise(F, wind_speed_kn=15.0, wind_model='flat50')
        assert np.allclose(w.wind, 50.0) and w.models['wind'] == 'flat50'
    finally:
        del pkg.WIND_MODELS['flat50']
    assert hasattr(pkg, 'NoiseComponents')


def test_coates_alternatives_registered():
    assert 'coates' in N.WIND_MODELS
    assert 'coates' in N.SHIPPING_MODELS


def test_coates_wind_plausible_and_differs():
    base = N.WenzNoise(F, wind_speed_kn=19.4)                       # ~10 m/s
    coates = N.WenzNoise(F, wind_speed_kn=19.4, wind_model='coates')
    assert coates.models['wind'] == 'coates'
    assert np.all(np.isfinite(coates.wind[np.isfinite(coates.wind)]))
    # Coates wind @1 kHz, 10 m/s ≈ 50 + 7.5*sqrt(10) - 40*log10(1.4) ≈ 68 dB
    w1k = N.WIND_MODELS['coates'](np.array([1000.0]), wind_speed_kn=19.4)[0]
    assert 60.0 < w1k < 75.0
    assert not np.array_equal(coates.wind, base.wind)   # genuinely different model


def test_total_finite_when_wind_zero_midband():
    # With wind=0 in the 3.7-5.6 kHz gap (turbulence negative, thermal
    # negative, shipping/rain off) nothing may floor a component to -inf, which
    # would drive the incoherent total to -inf. The real levels keep it finite.
    f = np.array([4000.0, 5000.0])
    w = N.WenzNoise(f, wind_speed_kn=0.0, shipping_level='no', rain_rate='no')
    assert np.all(np.isfinite(w.total))


def test_negative_wind_speed_rejected():
    # N8: uniform guard so the Coates √(wind) model can't silently produce NaN.
    with pytest.raises(ConfigurationError):
        N.WenzNoise(F, wind_speed_kn=-1.0)


def test_custom_submodel_errors_are_typed():
    # N9: a custom callable that raises, or returns a wrong-shaped array, is
    # surfaced as a typed ConfigurationError, not a raw TypeError/broadcast error.
    with pytest.raises(ConfigurationError):
        N.WenzNoise(F, wind_speed_kn=15.0, wind_model=lambda f, **k: 1 / 0)
    with pytest.raises(ConfigurationError):
        N.WenzNoise(F, wind_speed_kn=15.0, wind_model=lambda f, **k: np.zeros(f.size + 1))


def test_coates_shipping_silent_and_activity_order():
    assert np.all(N.SHIPPING_MODELS['coates'](
        F, shipping_level='no') == -np.inf)
    lo = N.SHIPPING_MODELS['coates'](np.array([100.0]), shipping_level='low')[0]
    hi = N.SHIPPING_MODELS['coates'](np.array([100.0]), shipping_level='high')[0]
    assert hi > lo                                     # more traffic → louder


def test_rain_follows_drdc_table_2():
    """DRDC Table 2 (§2.4): ``NL_rain = r0 + r1*f + r2*f^2 + r3*f^3`` with f in
    kHz, and above 7 kHz a constant slope ``m_0 = s_2*(0.1/log10 2)``, ``s_2 =
    -5`` — the same extension §2.3 applies to wind above 2 kHz."""
    table = {                    # rain rate: (r0, r1, r2, r3)
        'light':     (51.0769, 1.4687, -0.5232, 0.0335),   # 1 mm/h
        'moderate':  (61.5358, 1.0147, -0.4255, 0.0277),   # 5 mm/h
        'heavy':     (65.1107, 0.8226, -0.3825, 0.0251),   # 10 mm/h
        'veryheavy': (74.3464, 1.0131, -0.4258, 0.0277),   # 100 mm/h
    }
    fk = np.array([0.5, 1.0, 3.0, 7.0])
    for rate, (r0, r1, r2, r3) in table.items():
        out = N.RAIN_MODELS['torres_costa'](fk * 1000.0, rain_rate=rate)
        assert np.allclose(out, r0 + r1 * fk + r2 * fk ** 2 + r3 * fk ** 3)

    # Above 7 kHz: -5 dB/octave, anchored on the cubic's value at 7 kHz.
    f = np.array([7000.0, 14000.0])
    hi = N.RAIN_MODELS['torres_costa'](f, rain_rate='heavy')
    assert np.isclose(hi[1] - hi[0], -5.0)


def test_shipping_follows_drdc_equations_5_to_7():
    """DRDC eq. (5): ``76 - 20(log10 f - log10 c1)^2 + 5(c2-4)``, with
    ``c1`` = 30 deep / 65 shallow and ``c2`` = 1/4/7 for low/medium/high."""
    f = np.array([10.0, 100.0, 1000.0])
    for level, c2 in (('low', 1), ('medium', 4), ('high', 7)):
        for depth, c1 in (('deep', 30), ('shallow', 65)):
            ref = 76 - 20 * (np.log10(f) - np.log10(c1)) ** 2 + 5 * (c2 - 4)
            got = N.SHIPPING_MODELS['wenz'](f, shipping_level=level,
                                            water_depth=depth)
            assert np.allclose(got, ref)


def test_wind_follows_drdc_annex_a_below_the_cutoff():
    """Transcription of DRDC eqs. (8)-(16): ``f0 = 770 - 100 log10 u``,
    ``L0 = c0 + 20 log10 u - 17 log10(f0/770)`` with ``c0`` = 42 deep / 45
    shallow, the two branches ``L1``/``L2`` at ``s1 = 1.5`` / ``s2 = -5``
    dB/octave, melded with exponent ``a = -25``. The melding exponent applied
    here is ``1/a``; the ``-1/a`` printed in eq. (13)/(20) is a sign typo,
    since it makes the branch above f0 rise where eq. (15) defines it to fall
    at ``s2 = -5`` dB/octave. Below the cutoff the Annex A.2 listing agrees;
    above it, it does not — see
    :func:`test_wind_above_the_cutoff_matches_drdc_equations_18_19`."""
    f = np.array([10.0, 100.0, 500.0, 1000.0, 2000.0])
    f_wind, s1w, s2w, a = 2000.0, 1.5, -5.0, -25
    for u in (5.0, 10.0, 20.0, 35.0):
        for depth in ('deep', 'shallow'):
            cst = 45 if depth == 'shallow' else 42
            f0w = 770 - 100 * np.log10(u)
            l0w = cst + 20 * np.log10(u) - 17 * np.log10(f0w / 770)
            l1w = l0w + (s1w / np.log10(2)) * np.log10(f / f0w)
            l2w = l0w + (s2w / np.log10(2)) * np.log10(f / f0w)
            lw = l1w * (1 + (l1w / l2w) ** (-a)) ** (1 / a)
            assert np.allclose(N.compute_windnoise(f, u, depth),
                               10 * np.log10(10 ** (lw / 10)))
    assert f_wind == 2000.0


def test_wind_above_the_cutoff_does_not_depend_on_the_frequency_grid():
    """DRDC eq. (18)-(19) anchor the >2 kHz extension on the level at 2000 Hz
    itself: ``K = Lw,2000 - m0*(10log10 2000)``, ``Lw = K + m0*10log10 f``.
    uacpy implements that, so the result cannot depend on the caller's grid;
    the Annex A.2 listing instead anchors on the last in-grid sample below the
    cutoff, which makes it grid-dependent."""
    coarse = N.compute_windnoise(np.array([100., 1000., 4000.]), 15.0, 'deep')
    fine = N.compute_windnoise(np.array([100., 1999., 4000.]), 15.0, 'deep')
    assert coarse[-1] == pytest.approx(fine[-1])


def test_coates_alternatives_follow_stojanovic():
    """Stojanović's standard UW-comms ambient-noise set (after Coates 1989),
    with ``f`` in kHz, ``w`` in m/s and shipping activity ``s`` in [0, 1]:
    wind ``50 + 7.5*sqrt(w) + 20log10 f - 40log10(f+0.4)`` and shipping
    ``40 + 20(s-0.5) + 26log10 f - 60log10(f+0.03)``."""
    f = np.array([100.0, 1000.0, 10000.0])
    fk = f / 1000.0

    for u_kn in (10.0, 19.4, 30.0):
        w = u_kn / 1.9438445                      # the model wants m/s
        ref = (50.0 + 7.5 * np.sqrt(w) + 20.0 * np.log10(fk)
               - 40.0 * np.log10(fk + 0.4))
        assert np.allclose(N.WIND_MODELS['coates'](f, wind_speed_kn=u_kn), ref)

    for level, s in (('low', 0.0), ('medium', 0.5), ('high', 1.0)):
        ref = (40.0 + 20.0 * (s - 0.5) + 26.0 * np.log10(fk)
               - 60.0 * np.log10(fk + 0.03))
        assert np.allclose(N.SHIPPING_MODELS['coates'](f, shipping_level=level),
                           ref)


def test_wind_model_pair_delta_at_10_khz():
    """Coates minus Merklinger wind at 10 kHz, 25 kn, deep water: the two
    closed forms give 56.2155 dB (Coates, ``w = 25/1.9438445`` m/s) and
    51.4501 dB (DRDC eq. 18-19 from the 2 kHz anchor), a +4.765 dB delta —
    the guide §5's "at 25 kn Coates is 4.77 dB louder at 10 kHz". 10 kHz is
    not where the pair separates most; see
    ``test_the_wind_pair_separates_most_near_4_khz``."""
    f = np.array([10000.0])
    coates = N.WIND_MODELS['coates'](f, wind_speed_kn=25.0)[0]
    merk = N.WIND_MODELS['merklinger'](f, wind_speed_kn=25.0,
                                       water_depth='deep')[0]
    assert coates == pytest.approx(56.2155, abs=1e-3)
    assert merk == pytest.approx(51.4501, abs=1e-3)
    assert coates - merk == pytest.approx(4.7654, abs=1e-3)


def test_shipping_model_pair_delta_at_the_wenz_peak():
    """Wenz minus Coates shipping at the 30 Hz deep-water hump, 'high': the
    Wenz fit peaks at exactly 76 + 5·(7−4) = 91.0 dB while Coates gives
    ``50 + 26·log10(0.03) − 60·log10(0.06)`` = 83.7161 dB — the guide §5's
    7.3 dB disagreement."""
    f = np.array([30.0])
    wenz = N.SHIPPING_MODELS['wenz'](f, shipping_level='high',
                                     water_depth='deep')[0]
    coates = N.SHIPPING_MODELS['coates'](f, shipping_level='high')[0]
    assert wenz == pytest.approx(91.0, abs=1e-9)
    assert coates == pytest.approx(83.7161, abs=1e-3)
    assert wenz - coates == pytest.approx(7.2839, abs=1e-3)


def test_wind_above_the_cutoff_matches_drdc_equations_18_19():
    """``K = Lw,2000 - m0*(10 log10 2000)`` then ``Lw = K + m0*10 log10 f``,
    with ``m0 = s2*(0.1/log10 2)`` from eq. (17)."""
    m0 = -5.0 * (0.1 / np.log10(2))
    f = np.array([2500.0, 4000.0, 8000.0, 20000.0])
    for u in (5.0, 15.0, 30.0):
        for depth in ('deep', 'shallow'):
            lw2000 = N.compute_windnoise(np.array([2000.0]), u, depth)[0]
            spec = (lw2000 - m0 * 10 * np.log10(2000.0)) + m0 * 10 * np.log10(f)
            got = N.compute_windnoise(np.concatenate(([2000.0], f)), u, depth)[1:]
            assert np.allclose(got, spec)


class TestEveryRegistryEntryTakesTheSameFrequencyArgument:
    """``WenzNoise``'s docstring advertises every registry entry as
    ``model(frequencies, *, wind_speed_kn, water_depth, shipping_level,
    rain_rate, **_)`` and all five registries are in ``uacpy.noise.__all__``,
    so a caller may hold one and call it directly. Three spellings of the same
    frequencies must reach the same answer."""

    KW = dict(wind_speed_kn=10.0, water_depth='deep',
              shipping_level='medium', rain_rate='moderate')

    @staticmethod
    def _entries():
        from uacpy.noise import (RAIN_MODELS, SHIPPING_MODELS,
                                 THERMAL_MODELS, TURBULENCE_MODELS,
                                 WIND_MODELS)
        out = []
        for family, registry in (('wind', WIND_MODELS),
                                 ('shipping', SHIPPING_MODELS),
                                 ('rain', RAIN_MODELS),
                                 ('thermal', THERMAL_MODELS),
                                 ('turbulence', TURBULENCE_MODELS)):
            for name, fn in registry.items():
                out.append((f'{family}/{name}', fn))
        return out

    @pytest.mark.parametrize('label, model',
                             _entries.__func__(),
                             ids=[label for label, _ in _entries.__func__()])
    def test_a_list_and_an_ndarray_give_the_same_answer(self, label, model):
        want = model(np.array([100.0, 1000.0]), **self.KW)
        got = model([100.0, 1000.0], **self.KW)
        assert np.shape(got) == np.shape(want) == (2,)
        np.testing.assert_allclose(got, want)

    @pytest.mark.parametrize('label, model',
                             _entries.__func__(),
                             ids=[label for label, _ in _entries.__func__()])
    def test_a_scalar_gives_a_length_one_array(self, label, model):
        got = model(100.0, **self.KW)
        assert np.shape(got) == (1,)
        np.testing.assert_allclose(
            got, model(np.array([100.0]), **self.KW))

    def test_the_rain_model_takes_a_scalar_above_its_own_fit_limit(self):
        """The specific face: above 7 kHz the model indexes ``out[above]``,
        which a 0-d result met as ``'float' object is not subscriptable`` —
        so the failure depended on the *value*, not only on the type."""
        from uacpy.noise import RAIN_MODELS
        model = RAIN_MODELS['torres_costa']
        below = float(model(6999.0, **self.KW)[0])
        above = float(model(7001.0, **self.KW)[0])
        assert np.isfinite(below) and np.isfinite(above)
        assert abs(below - above) < 0.1     # continuous across the meld

    def test_a_two_dimensional_frequency_grid_is_refused_by_name(self):
        from uacpy.noise import THERMAL_MODELS
        with pytest.raises(ConfigurationError, match='frequencies'):
            THERMAL_MODELS['mellen'](np.ones((2, 2)), **self.KW)

    def test_wenznoise_assembles_a_finite_spectrum(self):
        """The negative control: the public wrapper already converted its
        input, so the assembled spectrum must be untouched."""
        from uacpy.noise import WenzNoise
        got = WenzNoise(np.array([100.0, 1000.0, 10000.0]),
                        wind_speed_kn=10.0).as_psd()
        assert np.shape(np.asarray(got)) == (3,)
        assert np.all(np.isfinite(np.asarray(got)))


class TestAMisspeltRegistryKeyIsATypedError:
    def test_shipping_and_rain_levels(self):
        from uacpy.noise.ambient import _shipping_wenz, _shipping_coates, _rain_torres_costa
        from uacpy.core.exceptions import ConfigurationError
        f = np.array([100.0, 1000.0])
        with pytest.raises(ConfigurationError, match='shipping_level'):
            _shipping_wenz(f, shipping_level='moderat', water_depth='deep')
        with pytest.raises(ConfigurationError, match='shipping_level'):
            _shipping_coates(f, shipping_level='moderat')
        with pytest.raises(ConfigurationError, match='rain_rate'):
            _rain_torres_costa(f, rain_rate='heavvy')


def test_the_wind_pair_separates_most_near_4_khz():
    """The guide called 10 kHz the widest gap. Over its own 10 Hz - 100 kHz
    grid the maximum is 5.14 dB near 4.3 kHz, and 10 kHz sits below it at
    4.77 dB. Coates is also the *quieter* of the two below ~66 Hz, by 10.1 dB
    at 10 Hz, which "up to 4.8 dB louder" did not say."""
    f = np.logspace(1.0, 5.0, 800)
    delta = (N.WIND_MODELS['coates'](f, wind_speed_kn=25.0)
             - N.WIND_MODELS['merklinger'](f, wind_speed_kn=25.0,
                                           water_depth='deep'))
    peak = int(np.argmax(delta))
    assert delta[peak] == pytest.approx(5.144, abs=5e-3)
    assert 4.0e3 < f[peak] < 4.6e3
    assert delta[peak] > np.interp(1.0e4, f, delta)      # 10 kHz is not the max
    assert delta[0] == pytest.approx(-10.09, abs=0.01)   # Coates quieter at 10 Hz
    assert f[delta < 0].max() == pytest.approx(66.0, rel=0.02)


def test_the_drdc_report_is_cited_by_its_own_year():
    """The module header and the report itself say DRDC-RDDC-2022-D051, May
    2022; three sites cited it as 2018."""
    import inspect
    src = inspect.getsource(N)
    assert 'Tollefsen & Pecknold (2018)' not in src
    assert 'Tollefsen & Pecknold 2018' not in src
    assert src.count('2022') >= 4


class TestKnudsenMatchesThePublishedCurves:
    """``44 + 20·log10(U_kn) - 17·log10(f_kHz)`` against the 1948 figure.

    The expected side is Knudsen, Alford & Emling (1948) Fig. 4 Part C read
    off the page — the overall 0.1-10 kc level of each of the seven curves —
    together with the two statements the paper makes in its own text: a
    -5 dB/octave spectrum slope, and levels referred to 0.0002 dyne/cm²
    (= 20 µPa, so +26.0 dB to reach dB re 1 µPa²/Hz). Nothing here calls
    into the model to build its own expectation.

    The model is a straight-line closed form and the curves are not quite
    straight in log10(U), so the test is that they agree inside the scatter
    the paper reports for its own observations ("of the order of 4 to 5 db")
    — not that they agree exactly. Demanding exactness would be demanding
    something the data does not support.

    Hildebrand et al. (2021) Table IV gives a third party's reading of the
    same figure, which bounds the risk that it was misread once here.
    """

    #: Fig. 4 Part C legend: overall 0.1-10 kc level per curve, keyed by
    #: wind force, in the paper's own dB re 0.0002 dyne/cm². Seven curves;
    #: the set skips force 6 (it plots sea states 0, ½, 1, 2, 3, 4 and 6).
    FIG4_OVERALL_DB = {0: 57.0, 1: 63.0, 2: 68.0, 3: 74.0, 4: 77.0,
                       5: 79.0, 7: 83.0}
    SLOPE_DB_PER_OCTAVE = 5.0
    BAND_HZ = (100.0, 10000.0)
    REF_SHIFT_DB = 20.0 * np.log10(20.0)       # 0.0002 dyne/cm² = 20 µPa

    #: Hildebrand et al. (2021) Table IV, "Knudsen (1948)" row: 1 kHz
    #: spectrum level, dB re 1 µPa²/Hz, by Beaufort force. The paper states
    #: these carry 0.5 dB precision, "due to the difficulty of estimation
    #: from published graphics".
    HILDEBRAND_TABLE_IV = {1: 51.0, 2: 55.5, 3: 61.5, 4: 64.5, 5: 66.5,
                           7: 71.0}

    @classmethod
    def _published_1k_levels(cls):
        """Fig. 4's curves as 1 kHz spectrum levels, dB re 1 µPa²/Hz."""
        a = (cls.SLOPE_DB_PER_OCTAVE / np.log10(2.0)) / 10.0
        f1, f2 = cls.BAND_HZ
        band = 10.0 * np.log10(1000.0 ** a / (a - 1.0)
                               * (f1 ** (1.0 - a) - f2 ** (1.0 - a)))
        return {force: lvl - band + cls.REF_SHIFT_DB
                for force, lvl in cls.FIG4_OVERALL_DB.items()}

    def test_the_two_readings_of_the_figure_agree(self):
        """This file's reading against Hildebrand's, before either is used
        to judge the model. Without this, a misread figure would be checked
        against itself."""
        mine = self._published_1k_levels()
        for force, want in self.HILDEBRAND_TABLE_IV.items():
            assert abs(mine[force] - want) <= 0.6, (
                f'force {force}: this file {mine[force]:.1f}, '
                f'Hildebrand Table IV {want:.1f}')

    def test_the_slope_is_the_paper_s_five_db_per_octave(self):
        """-17 dB/decade is -5 dB/octave rounded; the gap is what rounding
        costs, 0.12 dB per octave."""
        f = np.array([500.0, 1000.0, 2000.0, 4000.0])
        steps = np.diff(N.WIND_MODELS['knudsen'](f, wind_speed_kn=15.0))
        assert np.allclose(steps, -5.0, atol=0.15), steps

    def test_it_sits_inside_the_scatter_of_the_curves_it_fits(self):
        from uacpy.core.units import beaufort_to_wind_speed
        published = self._published_1k_levels()
        diffs = []
        for force in sorted(self.HILDEBRAND_TABLE_IV):
            u = beaufort_to_wind_speed(force)
            got = float(N.WIND_MODELS['knudsen'](np.array([1000.0]),
                                                 wind_speed_kn=u)[0])
            diffs.append(got - published[force])
        rms = float(np.sqrt(np.mean(np.square(diffs))))
        assert rms < N.KNUDSEN_UNCERTAINTY_DB, (rms, diffs)

    def test_the_constant_is_the_calm_curve(self):
        """At 1 knot and 1 kHz the formula is its constant, 44 dB, and
        Fig. 4's force-0 curve lands within a decibel of it."""
        got = float(N.WIND_MODELS['knudsen'](np.array([1000.0]),
                                             wind_speed_kn=1.0)[0])
        assert got == pytest.approx(44.0)
        assert abs(self._published_1k_levels()[0] - 44.0) < 1.0

    def test_it_is_the_stated_formula(self):
        f = np.array([250.0, 1000.0, 8000.0])
        got = N.WIND_MODELS['knudsen'](f, wind_speed_kn=12.0)
        want = 44.0 + 20.0 * np.log10(12.0) - 17.0 * np.log10(f / 1000.0)
        assert np.allclose(got, want)

    def test_zero_wind_switches_the_source_off(self):
        lvl = N.WIND_MODELS['knudsen'](np.array([1e3]), wind_speed_kn=0.0)
        assert np.all(np.isneginf(lvl))

    def test_the_curves_are_parallel_as_knudsen_drew_them(self):
        """His curves shift with wind by the same amount at every
        frequency, which is the property the straight-line form keeps."""
        f = np.array([200.0, 1000.0, 5000.0])
        a = N.WIND_MODELS['knudsen'](f, wind_speed_kn=5.0)
        b = N.WIND_MODELS['knudsen'](f, wind_speed_kn=25.0)
        assert np.allclose(b - a, (b - a)[0])


class TestAnEnvironmentKnobNoSubmodelReadsIsReported:
    """A setting that is accepted, validated, and then has no effect.

    ``water_depth='shallow'`` with the Coates submodels produced a spectrum
    identical to the deep-water one: Coates carries no depth term and
    swallowed the argument through ``**_``. Nothing here invents a depth
    dependence — the warning reports that the knob did nothing, which is
    the part the caller cannot otherwise see.
    """

    F = np.array([1000.0])

    def _build(self, **kw):
        return N.WenzNoise(self.F, wind_speed_kn=15.0, **kw)

    def test_it_warns_when_no_selected_submodel_carries_the_term(self):
        with pytest.warns(UserWarning, match="water_depth='shallow' was not"):
            self._build(water_depth='shallow', wind_model='coates',
                        shipping_model='coates')

    def test_the_warning_is_true_the_spectrum_really_is_unchanged(self):
        """Measured, not asserted: the two spectra are bit-identical."""
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            shallow = self._build(water_depth='shallow', wind_model='coates',
                                  shipping_model='coates').total
        deep = self._build(water_depth='deep', wind_model='coates',
                           shipping_model='coates').total
        assert np.array_equal(shallow, deep)

    def test_it_stays_silent_when_a_submodel_does_carry_the_term(self):
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            self._build(water_depth='shallow', wind_model='merklinger')
        assert not [c for c in caught if 'was not used' in str(c.message)]

    def test_it_stays_silent_when_the_knob_is_left_at_its_default(self):
        """Only a value the caller chose can be reported as ignored."""
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            self._build(water_depth='deep', wind_model='coates',
                        shipping_model='coates')
        assert not [c for c in caught if 'was not used' in str(c.message)]


class TestTheWindScaleConversions:
    """m/s to knots, and the Beaufort/sea-state scales, as callables.

    The factor and the Beaufort table both existed only as prose — the
    factor in two docstrings ("multiply the m/s returned here by 1.9438")
    and the table inside ``WenzNoise``'s docstring — while every fetcher
    returns m/s and every noise and scattering entry point takes knots.
    """

    def test_the_seam_these_close_costs_a_measured_5_7_db(self):
        """Why the conversion is a function: reading a m/s value as knots
        understates the total by this much, and nothing used to stop it."""
        from uacpy.core.units import ms_to_knots
        f = np.array([1000.0])
        kw = dict(water_depth='deep', shipping_level='medium')
        wrong = N.WenzNoise(f, wind_speed_kn=10.0, **kw).total[0]
        right = N.WenzNoise(f, wind_speed_kn=ms_to_knots(10.0), **kw).total[0]
        assert 5.5 < right - wrong < 6.0, right - wrong

    def test_knots_and_ms_round_trip(self):
        from uacpy.core.units import ms_to_knots, knots_to_ms
        for v in (0.0, 1.0, 7.3, 40.0):
            assert np.isclose(knots_to_ms(ms_to_knots(v)), v)

    def test_one_knot_is_the_standard_metres_per_second(self):
        from uacpy.core.units import knots_to_ms
        assert abs(float(knots_to_ms(1.0)) - 0.514444) < 1e-5

    @pytest.mark.parametrize('force, low, high', [
        (0, 0.0, 1.0), (3, 7.0, 10.0), (5, 17.0, 21.0), (8, 34.0, 40.0)])
    def test_a_force_maps_inside_its_own_band(self, force, low, high):
        from uacpy.core.units import beaufort_to_wind_speed
        u = beaufort_to_wind_speed(force)
        assert low <= u <= high

    def test_the_scale_agrees_with_the_table_in_wenznoise_s_docstring(self):
        """The table was prose; this pins the callable to it."""
        from uacpy.core.units import BEAUFORT_SCALE
        doc = N.WenzNoise.__doc__
        for force, (_sea_state, (low, high)) in BEAUFORT_SCALE.items():
            if force == 0:
                continue
            assert f'{low:g} – {high:g}' in doc, (force, low, high)

    def test_sea_state_six_resolves_to_force_eight(self):
        """Sea state 6 spans forces 7 and 8. Dahl et al. (2007) state the
        WMO correspondence as force 8, which is the branch taken."""
        from uacpy.core.units import (sea_state_to_wind_speed,
                                      beaufort_to_wind_speed)
        assert (sea_state_to_wind_speed(6)
                == beaufort_to_wind_speed(8))

    def test_wind_speed_to_beaufort_inverts_the_band(self):
        from uacpy.core.units import (beaufort_to_wind_speed,
                                      wind_speed_to_beaufort)
        for force in range(9):
            assert wind_speed_to_beaufort(
                beaufort_to_wind_speed(force)) == force

    def test_units_are_selectable_and_consistent(self):
        from uacpy.core.units import beaufort_to_wind_speed, knots_to_ms
        kn = beaufort_to_wind_speed(5, units='kn')
        ms = beaufort_to_wind_speed(5, units='m/s')
        assert np.isclose(knots_to_ms(kn), ms)

    @pytest.mark.parametrize('bad', [-1, 9, 'gale'])
    def test_a_force_off_the_table_is_refused(self, bad):
        from uacpy.core.units import beaufort_to_wind_speed
        with pytest.raises(ConfigurationError, match='force must be one of'):
            beaufort_to_wind_speed(bad)

    def test_an_unknown_unit_is_refused(self):
        from uacpy.core.units import beaufort_to_wind_speed
        with pytest.raises(ConfigurationError, match="units must be"):
            beaufort_to_wind_speed(5, units='mph')


class TestEverySubmodelUacpyShipsIsRecognisedAsItsOwn:
    """`_is_builtin_submodel` decides whose signature may be trusted.

    The "this knob was not used" warning only fires for submodels this
    package wrote, because a caller's own callable can read a parameter its
    signature never names — judging one by its signature produced a warning
    claiming a 26.8 dB difference did not exist.

    The test is `fn.__module__ == 'uacpy.noise.ambient'`, which is exact
    today because every built-in lives there. It is also silent if that
    stops being true: a submodel defined in or moved to another uacpy module
    becomes "custom", the warning quietly stops covering it, and no existing
    test notices. This is that notice.
    """

    def test_every_registered_submodel_is_recognised(self):
        registries = (N.WIND_MODELS, N.SHIPPING_MODELS, N.RAIN_MODELS,
                      N.THERMAL_MODELS, N.TURBULENCE_MODELS)
        shipped = {f'{r_name}[{key!r}]': fn
                   for r_name, registry in zip(
                       ('WIND', 'SHIPPING', 'RAIN', 'THERMAL', 'TURBULENCE'),
                       registries)
                   for key, fn in registry.items()}
        # Silence must mean "all recognised", never "none were found".
        assert len(shipped) >= 8, sorted(shipped)
        unrecognised = [name for name, fn in shipped.items()
                        if not N._is_builtin_submodel(fn)]
        assert not unrecognised, (
            f'submodel(s) uacpy ships that _is_builtin_submodel does not '
            f'recognise: {sorted(unrecognised)}. The unused-knob warning '
            f'has silently stopped covering them — either move them back '
            f'beside the registries or widen the test to the package.')

    def test_a_callers_own_function_is_not_recognised(self):
        """The other direction: whatever the caller passes stays theirs,
        including one they registered by name, which is a documented route
        (``WIND_MODELS['mine'] = fn``) and used to fail this."""
        def mine(f, **k):
            return np.zeros_like(f)
        assert not N._is_builtin_submodel(mine)
        N.WIND_MODELS['mine'] = mine
        try:
            assert not N._is_builtin_submodel(N.WIND_MODELS['mine'])
        finally:
            del N.WIND_MODELS['mine']

    def test_a_registered_custom_model_draws_no_false_warning(self):
        """End to end, on the shape that produced the false claim: a
        ``(f, **k)`` submodel that reads the knob through its kwargs."""
        import warnings as _w

        def depth_aware(f, **k):
            return np.full_like(
                f, 60.0 if k['water_depth'] == 'shallow' else 55.0)

        N.WIND_MODELS['depth_aware'] = depth_aware
        try:
            grid = np.array([1000.0])
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter('always')
                shallow = N.WenzNoise(grid, wind_speed_kn=15.0,
                                      water_depth='shallow',
                                      wind_model='depth_aware',
                                      shipping_model='coates').total[0]
            deep = N.WenzNoise(grid, wind_speed_kn=15.0, water_depth='deep',
                               wind_model='depth_aware',
                               shipping_model='coates').total[0]
        finally:
            del N.WIND_MODELS['depth_aware']
        # The model demonstrably reads the knob...
        assert abs(shallow - deep) > 1.0, (shallow, deep)
        # ...so claiming it did not would be false.
        assert not [c for c in caught if 'was not used' in str(c.message)]


class TestTheKnotHasOneValueAndOneHome:
    """One number, one declaration, and the SI definition as its source.

    It had three homes and two values — `1.9438445` twice and
    `1852.0/3600.0` once — which is the numeric case of the duplicate that
    `test_no_vocabulary_is_declared_twice` catches for strings and does not
    cover for constants. Nothing pinned any of them.
    """

    def test_it_is_the_si_definition(self):
        from uacpy.core.units import KNOTS_PER_M_PER_S, knots_to_ms
        # 1852 m in a nautical mile, 3600 s in an hour — not a rounded
        # literal, so the one value in the package is the exact one.
        assert KNOTS_PER_M_PER_S == 3600.0 / 1852.0
        assert float(knots_to_ms(1.0)) == pytest.approx(0.5144444444, abs=1e-10)

    def test_the_sonar_copy_derives_from_it_bit_for_bit(self):
        """`sonar.scattering` held its own `1852.0/3600.0`. It now reads the
        one home, and the reciprocal is bit-identical to what it replaced —
        which is not automatic, since 1/(a/b) need not equal b/a."""
        from uacpy.core.units import KNOTS_PER_M_PER_S
        from uacpy.sonar.scattering import _KNOT_TO_MS
        assert _KNOT_TO_MS == 1.0 / KNOTS_PER_M_PER_S
        assert _KNOT_TO_MS == 1852.0 / 3600.0

    def test_no_module_computes_with_a_copy_of_the_factor(self):
        """The shape the duplicate took: a hand-rounded copy of the factor.

        Parsed rather than grepped. The factor's digits legitimately appear
        in prose — a comment recalling what the docs used to say, another
        rescaling a different coefficient — and a text sweep flags those
        while missing a literal written any other way. ``ast`` sees only
        numbers the interpreter will use.
        """
        import ast
        import pathlib
        import uacpy
        root = pathlib.Path(uacpy.__file__).parent
        offenders, scanned = [], 0
        for path in sorted(root.rglob('*.py')):
            if 'third_party' in path.parts or 'tests' in path.parts:
                continue
            scanned += 1
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant)
                        and isinstance(node.value, float)
                        and abs(node.value - 1.9438444924406046) < 1e-4):
                    offenders.append(
                        f'{path.relative_to(root)}:{node.lineno}')
        # Silence must mean "none found", never "nothing was parsed".
        assert scanned > 50, scanned
        assert not offenders, (
            f'the knots factor computed from a copy at {offenders}; '
            f'uacpy.core.units.KNOTS_PER_M_PER_S is its one home.')


class TestTheForeignCarrierRefusalSaysWhatToDoInstead:
    """The refusal's remedy has to be true for every carrier it can name.

    It used to end "Use that result's own .plot()" — which three public
    carriers (`BeamformResult`, `NoiseComponents`, `Snapshots`) do not have,
    so the advice was wrong for exactly the results least likely to be
    recognised.
    """

    def test_it_does_not_promise_a_plot_that_may_not_exist(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from uacpy.acoustic_signal import snapshots
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.visualization import plot_spectrogram
        record = np.random.default_rng(0).standard_normal((512, 4))
        carrier = snapshots(record, 2000.0, 200.0, nperseg=256)
        assert not hasattr(carrier, 'plot')
        try:
            with pytest.raises(ConfigurationError) as caught:
                plot_spectrogram(carrier)
        finally:
            plt.close('all')
        message = str(caught.value)
        # It names what it got and what the plotter wants...
        assert 'Snapshots' in message and 'plot_spectrogram' in message
        # ...and offers a remedy that exists for a carrier without .plot().
        assert 'arrays' in message
