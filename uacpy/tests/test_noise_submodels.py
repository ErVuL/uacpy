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
from uacpy.noise import noise as N
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
    w = N.WenzNoise(F, wind_speed=15.0, rain_rate='heavy',
                    water_depth='deep', shipping_level='high')
    th = N.THERMAL_MODELS['mellen'](F)
    wi = N.WIND_MODELS['merklinger'](F, wind_speed=15.0, water_depth='deep')
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
    w = N.WenzNoise(F, 15.0)
    assert w.models == {'wind': 'merklinger', 'shipping': 'wenz',
                        'rain': 'torres_costa', 'thermal': 'mellen',
                        'turbulence': 'wenz'}


def test_string_selector_equals_default():
    a = N.WenzNoise(F, 15.0)
    b = N.WenzNoise(F, 15.0, wind_model='merklinger')
    assert np.array_equal(a.wind, b.wind)


def test_bad_selector_raises():
    with pytest.raises(ConfigurationError):
        N.WenzNoise(F, 15.0, wind_model='nope')


def test_custom_callable_changes_only_that_component():
    base = N.WenzNoise(F, 15.0)
    flat = N.WenzNoise(F, 15.0, wind_model=lambda f, **k: np.full_like(f, 50.0))
    assert np.allclose(flat.wind, 50.0)
    assert np.array_equal(flat.shipping, base.shipping)
    assert flat.models['wind'] == 'custom'


def test_components_namedtuple():
    w = N.WenzNoise(F, 15.0, rain_rate='light')
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
        w = N.WenzNoise(F, 15.0, wind_model='flat50')
        assert np.allclose(w.wind, 50.0) and w.models['wind'] == 'flat50'
    finally:
        del pkg.WIND_MODELS['flat50']
    assert hasattr(pkg, 'NoiseComponents')


def test_coates_alternatives_registered():
    assert 'coates' in N.WIND_MODELS
    assert 'coates' in N.SHIPPING_MODELS


def test_coates_wind_plausible_and_differs():
    base = N.WenzNoise(F, 19.4)                       # ~10 m/s
    coates = N.WenzNoise(F, 19.4, wind_model='coates')
    assert coates.models['wind'] == 'coates'
    assert np.all(np.isfinite(coates.wind[np.isfinite(coates.wind)]))
    # Coates wind @1 kHz, 10 m/s ≈ 50 + 7.5*sqrt(10) - 40*log10(1.4) ≈ 68 dB
    w1k = N.WIND_MODELS['coates'](np.array([1000.0]), wind_speed=19.4)[0]
    assert 60.0 < w1k < 75.0
    assert not np.array_equal(coates.wind, base.wind)   # genuinely different model


def test_total_finite_when_wind_zero_midband():
    # With wind=0 in the 3.7-5.6 kHz gap (turbulence negative, thermal
    # negative, shipping/rain off) nothing may floor a component to -inf, which
    # would drive the incoherent total to -inf. The real levels keep it finite.
    f = np.array([4000.0, 5000.0])
    w = N.WenzNoise(f, wind_speed=0.0, shipping_level='no', rain_rate='no')
    assert np.all(np.isfinite(w.total))


def test_negative_wind_speed_rejected():
    # N8: uniform guard so the Coates √(wind) model can't silently produce NaN.
    with pytest.raises(ConfigurationError):
        N.WenzNoise(F, wind_speed=-1.0)


def test_custom_submodel_errors_are_typed():
    # N9: a custom callable that raises, or returns a wrong-shaped array, is
    # surfaced as a typed ConfigurationError, not a raw TypeError/broadcast error.
    with pytest.raises(ConfigurationError):
        N.WenzNoise(F, 15.0, wind_model=lambda f, **k: 1 / 0)
    with pytest.raises(ConfigurationError):
        N.WenzNoise(F, 15.0, wind_model=lambda f, **k: np.zeros(f.size + 1))


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
        assert np.allclose(N.WIND_MODELS['coates'](f, wind_speed=u_kn), ref)

    for level, s in (('low', 0.0), ('medium', 0.5), ('high', 1.0)):
        ref = (40.0 + 20.0 * (s - 0.5) + 26.0 * np.log10(fk)
               - 60.0 * np.log10(fk + 0.03))
        assert np.allclose(N.SHIPPING_MODELS['coates'](f, shipping_level=level),
                           ref)


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
